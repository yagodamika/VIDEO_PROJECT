import json
from typing import Tuple, List

import cv2
import numpy as np
import GeodisTK
from tqdm import tqdm

from utils.video_utils import get_video_parameters, build_out_writer

WINDOW_PADDING = 5
MATTING_AREA_KERNEL_SIZE = 5
FG_PIXEL_DISTANCE_THRESH = 5
MATTING_SEARCH_WINDOW = 3
BG_SURE_THRESH = 0.1
FG_SURE_THRESH = 0.9


from matplotlib import pyplot as plt


def create_alpha_map(binary_img: np.ndarray, img_bgr: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """
    Function builds an alpha map for a given binary image using a rgb img
    We used the implementation as explained in class by alex
    :param binary_img: binary image output from the background segmentation phase
    :param img_bgr: bgr image from stabilized video
    :return: an alpha map with 1 in location of fg, 0 in location of bg and (0,1) for areas which are a blend and the bbox used for the crop
    """
    # Build a trimap
    kernel = np.ones((MATTING_AREA_KERNEL_SIZE, MATTING_AREA_KERNEL_SIZE))
    img_dil = cv2.dilate(binary_img.astype(np.uint8), kernel)
    img_ero = cv2.erode(binary_img.astype(np.uint8), kernel)
    alpha_area = img_dil - img_ero
    trimap = np.zeros_like(binary_img, dtype=float)
    trimap[binary_img < 127] = 0
    trimap[binary_img >= 127] = 1
    trimap[alpha_area > 0] = 0.5

    # We will be working on a crop of the area of the person to run faster
    x, y, w, h = cv2.boundingRect(trimap.astype(np.uint8))
    trimap_crop = trimap[y - WINDOW_PADDING: y + h + WINDOW_PADDING, x - WINDOW_PADDING:x + w + WINDOW_PADDING]
    img_rgb_crop = img_bgr[y - WINDOW_PADDING: y + h + WINDOW_PADDING, x - WINDOW_PADDING:x + w + WINDOW_PADDING]

    # Build FG and BG starting locations according to trimap
    fg_start = np.zeros_like(trimap_crop, dtype=np.uint8)
    fg_start[trimap_crop == 1] = 1
    bg_start = np.zeros_like(trimap_crop, dtype=np.uint8)
    bg_start[trimap_crop == 0] = 1

    # Build FG and BG maps
    fg_distance_map = 1 / (GeodisTK.geodesic2d_raster_scan(img_rgb_crop, fg_start, 1.0, 2) + 1e-6)
    bg_distance_map = 1 / (GeodisTK.geodesic2d_raster_scan(img_rgb_crop, bg_start, 1.0, 2) + 1e-6)
    alpha_crop = np.copy(trimap_crop)

    # Get alpha value
    alpha_crop[trimap_crop == 0.5] = fg_distance_map[trimap_crop == 0.5] / (bg_distance_map[trimap_crop == 0.5] + fg_distance_map[trimap_crop == 0.5])

    # Build final alpha map on top of trimap on the cropped location
    alpha_map = np.copy(trimap)
    alpha_map[y - WINDOW_PADDING: y + h + WINDOW_PADDING, x - WINDOW_PADDING:x + w + WINDOW_PADDING] = alpha_crop

    return alpha_map, (x, y, w, h)


def calculate_pixel_fg_value(bg_vals: np.ndarray, fg_vals: np.ndarray, pixel_value: np.ndarray, alpha: float) -> np.ndarray:
    """
    This implements part of the blending algorithm finding the fg pixel which minimizes the function (alpha*fg + (1-alpha)*bg) - pixel_value)
    :param bg_vals: pixel values of sure bg around  required pixel location
    :param fg_vals: pixel values of sure fg around  required pixel location
    :param pixel_value: value of original pixel
    :param alpha: alpha value for required pixel
    :return:
    """
    # We set the initial guess to the current pixel value although this doesn't affect the output
    closest_val_fg = pixel_value
    min_distance = 1000
    # Iterate over combinations of sure bg and fg values finding the combination minimizing the required function
    for bg_val in bg_vals:
        for fg_val in fg_vals:
            calc_distance = np.linalg.norm((alpha * fg_val + (1 - alpha) * bg_val) - pixel_value)
            if calc_distance < min_distance:
                closest_val_fg = fg_val
                min_distance = calc_distance
                # To speed up calculation, we allow the distance calculation below a certain thresh to be used as the fg pixel
                if calc_distance < FG_PIXEL_DISTANCE_THRESH:
                    return closest_val_fg

    return closest_val_fg


def get_matted_fg_background_color(img_bgr: np.ndarray, alpha_map: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Get the pixel values to be used for the area of the alpha maps considered as not sure. This is done using blending
    We use as input the bgr image and the background and using the computation of the fg pixel create the required blending
    :param img_bgr: original image
    :param alpha_map: the given alpha map from the geodesic distance calculation
    :param background: a given background
    :return: an array the same size of the original image with values to be used for the alpha area of the trimap
    """
    matted_fg_bg_color = np.zeros_like(img_bgr)
    # We only take care of positions which are above BG_SURE_THRESH and below FG_SURE_THRESH
    pos = (np.where((alpha_map > BG_SURE_THRESH) & (alpha_map < FG_SURE_THRESH)))
    for (i, j) in zip(pos[0], pos[1]):
        # Get a window to be used for finding the required fg pixel
        window = alpha_map[i - MATTING_SEARCH_WINDOW: i + MATTING_SEARCH_WINDOW + 1, j - MATTING_SEARCH_WINDOW: j + MATTING_SEARCH_WINDOW + 1]
        window_rgb = img_bgr[i - MATTING_SEARCH_WINDOW: i + MATTING_SEARCH_WINDOW + 1, j - MATTING_SEARCH_WINDOW: j + MATTING_SEARCH_WINDOW + 1]

        # We consider a sure fg or bg when the map is above 0.95 or below 0.05
        sure_fg = np.where(window >= 0.95)
        sure_fg_values = window_rgb[sure_fg]
        sure_bg = np.where(window <= 0.05)
        sure_bg_values = window_rgb[sure_bg]
        pixel_orig_value = img_bgr[i, j]

        # Get the estimated fg pixel value
        fg_pixel_value = calculate_pixel_fg_value(sure_bg_values, sure_fg_values, pixel_orig_value, alpha_map[i, j])
        matted_fg_bg_color[i, j] = (alpha_map[i, j][..., None] * fg_pixel_value) + ((1 - alpha_map[i, j][..., None]) * background[i, j])

    return matted_fg_bg_color


class MattingAndBBoxCreator:
    def __init__(self, video_cap_stabilized: cv2.VideoCapture, video_cap_binary: cv2.VideoCapture, background_path: str,
                 output_video_path_alpha: str, output_video_path_matting: str, output_video_path_final: str,
                 output_detections_json_path: str):
        self.cap_stabilized = video_cap_stabilized
        self.cap_binary = video_cap_binary

        self.video_params_stabilized = get_video_parameters(self.cap_stabilized)
        self.background = cv2.resize(cv2.imread(background_path), (self.video_params_stabilized['width'], self.video_params_stabilized['height']))
        self.out_writer_alpha = build_out_writer(output_video_path_alpha, self.video_params_stabilized)
        self.out_writer_matting = build_out_writer(output_video_path_matting, self.video_params_stabilized)
        self.out_writer_output = build_out_writer(output_video_path_final, self.video_params_stabilized)
        self.output_detections_json = output_detections_json_path

        self.bounding_boxes = np.zeros((self.video_params_stabilized['frame_count'], 4))

        self.run_matting_and_detection()
        self.write_bbox_to_json()

    @staticmethod
    def create_final_image(img_bgr: np.ndarray, alpha_map: np.ndarray, background: np.ndarray, matted_fg_bg_color: np.ndarray) -> np.ndarray:
        """
        Method creates the final image for a given frame using bgr image, alpha mao, background and the computed matted_fg_bg_color
        :param img_bgr: original image
        :param alpha_map: calculated alpha map
        :param background: given background image
        :param matted_fg_bg_color: calculated values for the "not sure" alpha values
        :return: the new image with the matted object
        """
        new_img = np.zeros_like(img_bgr)
        for i in range(3):
            new_img = ((1 - alpha_map)[..., None] * background) + (alpha_map[..., None] * img_bgr)
            new_img[matted_fg_bg_color != 0] = matted_fg_bg_color[matted_fg_bg_color != 0]
            new_img = np.clip(new_img, 0, 255).astype(np.uint8)

        return new_img

    @staticmethod
    def plot_bbox_on_image(final_image: np.ndarray, bbox_params: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Function plots the given bbox in format of x, y, w, h on image
        :param final_image: final image after matting
        :param bbox_params: box parameters
        :return: image with bbox
        """
        x, y, w, h = bbox_params
        start_point = (x, y)
        end_point = (x + w, y + h)
        color = (0, 255, 0)  # Green
        thickness = 2

        # Draw rectangle
        return cv2.rectangle(final_image, start_point, end_point, color, thickness)

    def run_matting_and_detection(self) -> None:
        """
        Method runs the matting algorithm
        :return: None
        """
        print("Alpha map, matting and bbox creation phase")
        for i in tqdm(range(0, self.video_params_stabilized['frame_count'])):

            # Get the stabilized color image and the matching binary image using background subtraction
            ret, img_bgr = self.cap_stabilized.read()
            ret, img_bin = self.cap_binary.read()
            img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
            img_bin[img_bin < 127] = 0
            img_bin[img_bin >= 127] = 255

            # Build the alpha map
            alpha_map, bbox_params = create_alpha_map(img_bin, img_bgr)
            self.bounding_boxes[i] = np.array(bbox_params)

            # Get the values of the areas which are not considered
            matted_fg_bg_color = get_matted_fg_background_color(img_bgr, alpha_map, self.background)

            # Build the matted image
            matted_img = self.create_final_image(img_bgr, alpha_map, self.background, matted_fg_bg_color)

            # Build the output image
            output_img = self.plot_bbox_on_image(np.copy(matted_img), bbox_params)

            # Write alpha map and matted image
            self.out_writer_alpha.write(cv2.cvtColor((alpha_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            self.out_writer_matting.write(matted_img)
            self.out_writer_output.write(output_img)

        # Release all writer resources
        self.out_writer_alpha.release()
        self.out_writer_matting.release()
        self.out_writer_output.release()

    def write_bbox_to_json(self) -> None:
        """
        Method writes output of bbox to json file
        :return: None
        """
        detection_output_dict = {str(frame_num + 1): self.bbox_params_to_required_format(params) for (frame_num, params) in enumerate(self.bounding_boxes)}
        with open(self.output_detections_json, 'w') as json_handler:
            json.dump(detection_output_dict, json_handler, indent=4)

    @staticmethod
    def bbox_params_to_required_format(params) -> List[int]:
        """
        Method truns bbox from top left corner to center format
        :param params: bbox params in top left corner format
        :return: list of params in center bbox format (enter_y, center_x, h/2, w/2)
        """
        w = params[2] // 2
        h = params[3] // 2
        x = params[0] + w
        y = params[1] + h

        return [y, x, h, w]

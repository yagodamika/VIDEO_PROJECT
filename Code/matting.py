import numpy as np
import cv2
from GeodisTK import geodesic2d_raster_scan
from tqdm import tqdm
from typing import Tuple
import json
from scipy.stats import gaussian_kde

PADDING = 5

from utils import parse_video, get_video_writer


def get_trimap(binary_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    This function builds a trimap
    for a given binary image.

    Args:
        binary_image: a binary image
        kernel_size: size of the kernel for dilation and
                    erosion to determine undecided area
    Return:
        A trimap - np array containing 1 for foreground,
        0 for background and 0.5 for undecided area
    """
    binary_image = binary_image.astype(np.uint8)

    # kernel used for determining the undecided area
    alpha_ker = np.ones((kernel_size, kernel_size))

    img_dilated = cv2.dilate(binary_image, alpha_ker)
    img_eroded = cv2.erode(binary_image, alpha_ker)

    undecided = np.subtract(img_dilated, img_eroded, dtype=np.int8)

    trimap = np.where(binary_image < 127, 0, 1).astype(float)
    trimap = np.where(undecided > 0, 0.5, trimap)

    return trimap


def get_interest_crop(array: np.ndarray, bounding_box: Tuple) -> np.ndarray:
    """
    This function returns a crop of area of interest
    of the trimap and its corresponding image.
    Args:
        trimap: a trimap
        image: the trimap's corresponding image
    Return:
        cut_img: a crop of the image
        cut_trimap: acrop of the trimap
        bounding_box: a tuple containing bounding box parameters
    """
    x, y, w, h = bounding_box
    p = PADDING
    cut = array[y-p:y+h+p, x-p:x+w+p]

    return cut


def get_geodesic_dist_maps(cut_luma: np.ndarray, fg_st: np.ndarray, bg_st: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        This function calculates geodesic distance maps for foreground and background using a cut
        of the image in luma format, foreground starting points and background starting points
        Args:
            cut_luma: a luma component of a cut of the original image
            fg_st: foreground starting points
            bg_st: background starting points
        Return:
            fg_d_map: forground distance map
            bg_d_map: background distance map
        """
    fg_d_map = geodesic2d_raster_scan(cut_luma, fg_st, 1.0, 2) + 1e-5
    bg_d_map = geodesic2d_raster_scan(cut_luma, bg_st, 1.0, 2) + 1e-5

    return fg_d_map, bg_d_map


def get_alpha_vals_for_crop(cut_luma: np.ndarray, cut_trimap: np.ndarray, fg_d_map: np.ndarray, bg_d_map: np.ndarray,
                            power: int = -1) -> np.ndarray:
    """
   This function calculates alpha values for image crop
   using a formula presented in the lecture
   Args:
       cut_luma: a luma component of a cut of the original image
       cut_trimap: a cut of the trimap
       fg_d_map: forground distance map
       bg_d_map: background distance map
       power: a power used for calculation
   Return:
       alpha_cut: a cut of the alpha map
    """
    alpha_cut = np.copy(cut_trimap)
    mask = (cut_trimap == 0.5)
    fg_distances = fg_d_map[mask]
    bg_distances = bg_d_map[mask]

    bg_pdf = gaussian_kde(cut_luma[cut_trimap == 0])
    bg_probs = bg_pdf.evaluate(cut_luma[mask])
    fg_pdf = gaussian_kde(cut_luma[cut_trimap == 1])
    fg_probs = fg_pdf.evaluate(cut_luma[mask])

    w_b = bg_probs * (bg_distances ** power)
    w_f = fg_probs * (fg_distances ** power)
    alpha_values = w_f / (w_f + w_b)
    alpha_cut[mask] = alpha_values
    return alpha_cut


def get_alpha_map(binary_image: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    This function creates an alpha map.
    Args:
        binary_image: a binary image
        image: a bgr image corresponding to the alpha map
    Return:
        alpha_map: an alpha map
        bounding_box: the parameters of the area of interest
                       bounding box used to create the alpha map
    """
    trimap = get_trimap(binary_image)
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    luma_image = yuv_image[:, :, 0]

    # Identify a bounding rectangle that encloses the person
    bounding_box = cv2.boundingRect(trimap.astype(np.uint8))
    cut_img = get_interest_crop(image, bounding_box)
    cut_trimap = get_interest_crop(trimap, bounding_box)
    cut_luma = get_interest_crop(luma_image, bounding_box)

    # get foreground and background start places by trimap
    fg_st = np.ones_like(cut_trimap, dtype=np.uint8)
    fg_st[cut_trimap != 1] = 0
    bg_st = np.ones_like(cut_trimap, dtype=np.uint8)
    bg_st[cut_trimap != 0] = 0

    # get foreground and background geodesic distance maps
    fg_d_map, bg_d_map = get_geodesic_dist_maps(cut_luma, fg_st, bg_st)

    # get alpha values for crop
    alpha_cut = get_alpha_vals_for_crop(cut_luma, cut_trimap, fg_d_map, bg_d_map)

    # get alpha map
    x, y, w, h = bounding_box
    p = PADDING
    alpha_map = np.copy(trimap)
    alpha_map[y-p:y+h+p, x-p:x+w+p] = alpha_cut

    return alpha_map, bounding_box


def calc_fg_value(fg_vals: np.ndarray, bg_vals: np.ndarray, pixel_val: np.ndarray, alpha: float,
                  distance_threshold: int = 5) -> np.ndarray:
    """
    This function finds a foreground pixel value which minimizes
    an equation we seen in the lecture :
    ||(alpha*fg + (1-alpha)*bg) - pixel_value)||^2

    Args:
        fg_vals: values of foreground pixels in the window around the pixel we are looking at
        bg_vals: values of background pixels in the window around the pixel we are looking at
        pixel_val: value of the pixel we are looking at
        alpha: alpha value of the pixel we are looking at
        distance_threshold: if we find a distance below this threshold
                            we return it for efficiency
    Return:
        result_val: pixel value that minimizes the equation above

    """
    result_val = pixel_val
    min_dist = np.inf
    for fg in fg_vals:
        for bg in bg_vals:
            func = (alpha * fg + (1 - alpha) * bg)
            dist = np.linalg.norm(func - pixel_val)

            if dist < min_dist:
                result_val = fg
                min_dist = dist
            if dist < distance_threshold:
                return result_val
    return result_val


def crop_window(input: np.ndarray, x: int, y: int, w: int) -> np.ndarray:
    """
    This function crops a np array according to given parameters

    Args:
        input: input np array
        x: x value to crop around
        y: y value to crop around
        w: width of crop

    Return:
        crop: a crop of the input
    """
    crop = input[x-w:x+w+1, y-w:y+w+1]
    return crop


def get_confident_vals(window_alpha: np.ndarray, window_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
     Identifies confident foreground and background based on the alpha map values in a window

    Args:
        window_alpha: a window cropped from alpha map
        window_image: a window cropped from the original image

    Return:
        c_fg_vals: values of confident fg pixels in the window
        c_bg_vals: values of confident bg pixels in the window
    """

    up_threshold = 0.95
    down_threshold = 0.05

    c_fg_idxs = np.argwhere(window_alpha >= up_threshold)
    c_fg_vals = window_image[c_fg_idxs[:, 0], c_fg_idxs[:, 1]]
    c_bg_idxs = np.argwhere(window_alpha <= down_threshold)
    c_bg_vals = window_image[c_bg_idxs[:, 0], c_bg_idxs[:, 1]]

    return c_fg_vals, c_bg_vals


def get_undecided_vals(img: np.ndarray, alpha_map: np.ndarray, bg: np.ndarray, window_size: int) -> np.ndarray:
    """
    This function calculates the colors of the undecided area of the alpha map.

    Args:
        img: the original image
        alpha_map: the alpha map
        bg: background image
        window_size: a size of a window opened around undecided valued pixels for calculations

    Return:
        undecided_vals: color values for undecided values

    """
    w = window_size

    undecided_vals = np.zeros_like(img)
    condition = (alpha_map > 0.1) & (alpha_map < 0.9)
    positions = np.where(condition)

    for (x, y) in zip(positions[0], positions[1]):
        window_alpha = crop_window(alpha_map, x, y, w)
        window_image = crop_window(img, x, y, w)

        # Identify confident foreground and background based on the alpha map window values
        c_fg_vals, c_bg_vals = get_confident_vals(window_alpha, window_image)

        # Obtain estimated foreground pixel value
        alpha = alpha_map[x, y]
        original_pixel_val = img[x, y]
        fg_pixel_value = calc_fg_value(c_fg_vals, c_bg_vals, original_pixel_val, alpha)
        matted_color = (alpha_map[x, y][..., None] * fg_pixel_value) + ((1 - alpha[x, y][..., None]) * bg[x, y])

        undecided_vals[x, y] = matted_color

    return undecided_vals


def get_binary_img(binary_img: np.ndarray) -> np.ndarray:
    """
    This function formates the binary image with 255 and 0 values only

    Args:
        binary_img: the original binary image

    Return:
        binary_img: formatted binary image
    """
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    binary_img[binary_img >= 127] = 255
    binary_img[binary_img < 127] = 0
    return binary_img


class MattingMaker:
    def __init__(self, cap: cv2.VideoCapture, cap_binary: cv2.VideoCapture, background_img: np.ndarray,
                 output_path_alpha: str, output_path_matting: str, output_path_final: str,
                 detections_json_path: str):

        self.capture = cap
        self.cap_binary = cap_binary

        self.video_params = parse_video(self.capture)
        self.bg = cv2.resize(background_img, (self.video_params.frame_width, self.video_params.frame_height))

        self.out_writer_alpha = get_video_writer(output_path_alpha, self.video_params)
        self.out_writer_matting = get_video_writer(output_path_matting, self.video_params)
        self.out_writer_output = get_video_writer(output_path_final, self.video_params)
        self.detections_json = detections_json_path

    def apply(self):
        self.bboxs = np.zeros((self.video_params.frame_num, 4))

        self.matting_detection()
        self.write_bounding_box_json()

    @staticmethod
    def get_result_image(img: np.ndarray, alpha_map: np.ndarray, bg: np.ndarray,
                         undecided_vals: np.ndarray) -> np.ndarray:
        """
        Generates the result image by combining the original image, alpha map, background, and undecided values

        Args:
            img: the original image, in bgr format
            alpha_map: the alpha map
            bg: the background image
            undecided_vals: the values computed for the undecided area in the alpha map

        Return:
            result: the new image
        """
        result = np.zeros_like(img)
        alpha_map_expanded = np.expand_dims(alpha_map, axis=-1)
        undecided_idxs = (undecided_vals != 0)

        result = (alpha_map_expanded * img) + ((1 - alpha_map_expanded) * bg)
        result[undecided_idxs] = undecided_vals[undecided_idxs]
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)

    @staticmethod
    def draw_bbox(img: np.ndarray, box_parameters: Tuple[float, float, float, float]) -> np.ndarray:
        """
        This function draws a bounding box on the given image

        Args:
            img: matted image
            alpha_map: the alpha map
            bg: the background image
            undecided_vals: the values computed for the undecided area in the alpha map

        Return:
            result: the new image
        """
        x, y, w, h = box_parameters
        start_pt, end_pt = (x, y), (x + w, y + h)
        thickness = 2
        hue = (0, 255, 0)

        return cv2.rectangle(img, start_pt, end_pt, hue, thickness)

    def matting_detection(self) -> None:
        """
        This function performs matting and detection
        """
        for i in tqdm(range(0, self.video_params.frame_num)):
            _, img_bgr = self.capture.read()
            _, binary_img = self.cap_binary.read()

            # get binary image
            binary_img = get_binary_img(binary_img)

            # get an alpha map
            alpha_map, box_parameters = get_alpha_map(binary_img, img_bgr)

            # get the values of the areas which are not considered as bg or fg
            undecided_vals = get_undecided_vals(img_bgr, alpha_map, self.bg, window_size=3)

            # build the matted image
            matted = self.get_result_image(img_bgr, alpha_map, self.bg, undecided_vals)

            # get the image with a bounding box on it
            self.bboxs[i] = np.array(box_parameters)
            final_img = self.draw_bbox(np.copy(matted), box_parameters)

            # write alpha map, matted image and final image
            self.out_writer_alpha.write(cv2.cvtColor((alpha_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            self.out_writer_matting.write(matted)
            self.out_writer_output.write(final_img)

        self.out_writer_alpha.release()
        self.out_writer_matting.release()
        self.out_writer_output.release()

    def write_bounding_box_json(self) -> None:
        """
        This function writes bounding boxes parameters to a json file
        """
        dict = {}
        for (frame_idx, bbox_paramas) in enumerate(self.bboxs):
            w = bbox_paramas[2] // 2
            h = bbox_paramas[3] // 2
            x = bbox_paramas[0] + w
            y = bbox_paramas[1] + h
            frame_str = str(frame_idx + 1)
            dict[frame_str] = [y, x, h, w]

        with open(self.detections_json, 'w') as h:
            json.dump(dict, h, indent=4)


def paste_object_on_background(cap: cv2.VideoCapture, cap_binary: cv2.VideoCapture, bg_path: str,
                               output_path_alpha: str, output_path_matting: str, output_path_final: str,
                               detections_json_path: str):
    background_img = cv2.imread(bg_path)
    matting_maker = MattingMaker(cap, cap_binary, background_img, output_path_alpha, output_path_matting, output_path_final, detections_json_path)
    matting_maker.apply()
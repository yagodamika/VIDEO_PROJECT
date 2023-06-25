import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sklearn.neighbors import KernelDensity
import GeodisTK
from skimage.filters import threshold_multiotsu
from tqdm import tqdm

from utils.video_utils_ref import get_video_parameters, build_out_writer


KDE_TEMPLATE_SIZE = 100000
TEMPERATURE_FACTOR = 50
SEG_TIME_DIFF_THRESH = 8500
SEG_TIME_DIFF_RETRAIN_THRESH = 7000
PDF_HIST_LOW_THRESH = 0.002
PDF_HIST_HIGH_THRESH = 0.998
HIGH_PROB_FG_PIXELS_THRESH = 0.8

class MixturesOfGaussians:
    def __init__(self, num_frames: int, num_mixtures: int, var_threshold: int, is_forward: bool = True):
        self.is_forward = is_forward
        self.num_frames = num_frames

        self.mog = cv2.createBackgroundSubtractorMOG2(history=num_frames)
        self.mog.setNMixtures(num_mixtures)
        self.mog.setVarThreshold(var_threshold)

    def fit(self, capture: cv2.VideoCapture, nof_iterations: int) -> None:
        """
        Method fits the mog model to the video capture
        :param capture: video capture
        :param nof_iterations: number of iterations to run the fitting
        :return: None
        """

        for i in range(nof_iterations):
            # start capture from beginning
            capture.set(1, 0)

            for frame_idx in range(self.num_frames):
                if not self.is_forward:
                    capture.set(1, self.num_frames - frame_idx - 1)
                
                _, img = capture.read()
                self.mog.apply(img, learningRate=None)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Method applies the mog model to an image
        :param img: image to apply mog on
        :return: fg mask
        """

        return self.mog.apply(img, learningRate=0)

class BackgroundSubtractor:
    def __init__(self, video_cap_stabilized: cv2.VideoCapture,
                 output_video_path_binary: str, output_video_path_extracted: str):
        
        self.capture = video_cap_stabilized
        self.video_params = get_video_parameters(self.capture)
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.out_writer_binary = build_out_writer(output_video_path_binary, self.video_params, 0)
        self.out_writer_extracted = build_out_writer(output_video_path_extracted, self.video_params)

        self.forward_mog = MixturesOfGaussians(num_frames=self.num_frames, num_mixtures=5, var_threshold=4, is_forward=True)
        self.backward_mog = MixturesOfGaussians(num_frames=self.num_frames, num_mixtures=5, var_threshold=4, is_forward=False)
        self.forward_mog.fit(self.capture, 5)
        self.backward_mog.fit(self.capture, 5)

        self.accumulative_kde_template = None
        self.kde = None
        self.hist_min_val = 0
        self.hist_max_val = 255
        self.build_kde_and_hist_values(nof_frames_for_template=5)
        self.last_trained_frame = 5

        self.run_mog_on_movie_and_save_output()

        self.out_writer_binary.release()
        self.out_writer_extracted.release()

    @staticmethod
    def post_process_fg(fg, opening: bool = True, closing: bool = True) -> np.ndarray:
        """
        Method runs a post process on Bg which keeps only one largest component and runs opening and closing if required
        :param fg: fg mask
        :param opening: signals if to run morphological opening (default True)
        :param closing: signals if to run morphological closing (default True)
        :return: fg after post process
        """
        # Take only values with 255 (no shadows)
        fg[fg < 255] = 0

        # As we know this is a human walking, and we know th camera point of view, clear all pixels on top
        h, w = fg.shape
        fg[0:h//10, :] = 0

        # Build connected components, we will take the largest component as it's our fg object
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=4)
        sizes = stats[:, -1]

        max_label = int(np.argmax(sizes[1:])) + 1
        fg = np.zeros_like(output)
        fg[output == max_label] = 255

        # Run morphology of opening and closing to clean noise
        if opening:
            kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kernel_opening)
        if closing:
            kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel_closing)

        # Fix to binary values
        fg[fg < 127] = 0
        fg[fg >= 127] = 255

        return fg

    def build_kde_and_hist_values(self, nof_frames_for_template: int) -> None:
        """
        Method builds a kde object dnd a min max value from GS histogram used for choosing pixels to use or filter
        :param num_of_frames_for_template: number of frames used for template of building KDE and histogram
        :return: None
        """
        # start capture from beginning
        self.capture.set(1, 0)

        self.accumulative_kde_template = np.empty([0, 3])
        accumulative_gray_template = np.empty([0, 1])

        # Iterate over num_of_frames_for_template frames and build kde bgr and histogram grayscale template
        for frame in range(nof_frames_for_template):
            ret, img = self.capture.read()
            fg = self.forward_mog.apply(img)
            fg = self.post_process_fg(fg)
            frame_template = img[fg == 255]
            img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frame_template_gs = img_gs[fg == 255]

            # BGR KDE
            self.accumulative_kde_template = np.append(self.accumulative_kde_template, frame_template, axis=0)

            # GS histogram
            accumulative_gray_template = np.append(accumulative_gray_template, np.expand_dims(frame_template_gs, 1), axis=0)

        # KDE fitting
        self.kde = KernelDensity(bandwidth=0.3, kernel='gaussian', atol=0.000000005)
        self.kde.fit(self.accumulative_kde_template)

        # GS unused values
        probs, bins = np.histogram(accumulative_gray_template.flatten(), bins=30, density=True)
        dx = bins[1] - bins[0]
        pdf = np.cumsum(probs) * dx

        # We set a threshold for cleaning fg according to template PDF. Note we clear from edges and not from middle
        if len(np.where(pdf < PDF_HIST_LOW_THRESH)[0]) > 0:
            self.hist_min_val = bins[np.max(np.where(pdf < PDF_HIST_LOW_THRESH))]
        else:
            self.hist_min_val = 0

        if len(np.where(pdf > PDF_HIST_HIGH_THRESH)[0]) > 0:
            self.hist_max_val = bins[np.min(np.where(pdf > PDF_HIST_HIGH_THRESH))]
        else:
            self.hist_max_val = 255

    def retrain_kde(self, additional_pixels: np.ndarray) -> None:
        """
        Method retrains KDE when required
        :param additional_pixels: a set of pixels to add to self.accumulative_kde_template
        :return: None
        """

        # Add new pixels to template
        self.accumulative_kde_template = np.append(self.accumulative_kde_template, additional_pixels, axis=0)

        # If len of KDE is above KDE_TEMPLATE_SIZE keep only size
        if len(self.accumulative_kde_template) > KDE_TEMPLATE_SIZE:
            self.accumulative_kde_template = self.accumulative_kde_template[-KDE_TEMPLATE_SIZE:]
        self.kde = KernelDensity(bandwidth=0.3, kernel='gaussian', atol=0.000000005)
        self.kde.fit(self.accumulative_kde_template)

    def run_mog_on_movie_and_save_output(self) -> None:
        """
        Method runs background subtraction on entire movie frame by frame
        :return: None
        """
        # Start by setting video to first frame
        self.capture.set(1, 0)

        # prev_fg is used to calculate the difference from prev frame used to "signal" problematic frames
        prev_fg = None

        # Iterate over all frames and run mog
        print("Background Subtraction phase")
        for frame in tqdm(range(self.num_frames)):
            ret, img = self.capture.read()
            # Split use of mog by part of movie
            if frame < self.num_frames//2:
                fg = self.forward_mog.apply(img)
            else:
                fg = self.backward_mog.apply(img)

            # Post process fg
            fg = self.post_process_fg(fg)
            # We save a copy of fg at this point to allow filling holes later for wrong removed pixels in histogram removal
            fg_before_histogram_removal = np.copy(fg)

            # Remove the unused values according to hist
            img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fg[img_gs < self.hist_min_val] = 0
            fg[img_gs > self.hist_max_val] = 0

            # We save a copy after histogram removal for further computation on big diff frames
            fg_after_histogram_removal = np.copy(fg)

            # From the 5th frame on (as the first 5 were used for templating) we check the time derivative
            # change in the direction of the movement to "signal" frames which might contain noise
            if frame > 5:
                # Get the time diff of fg in the direction of the movement
                seg_time_diff = fg - prev_fg
                seg_time_diff = seg_time_diff * (seg_time_diff > 0)
                # If diff is large, use kde refinement to clear noise
                if (np.sum(seg_time_diff) / 255) > SEG_TIME_DIFF_THRESH:
                    fg = self.refine_fg_using_kde(fg_after_histogram_removal, img)

                # If change is small and kde hasn't been retrained in last 5 frames, retrain kde
                elif ((np.sum(seg_time_diff) / 255) < SEG_TIME_DIFF_RETRAIN_THRESH) and (frame - self.last_trained_frame >= 5):
                    self.retrain_kde(img[fg > 0])
                    self.last_trained_frame = frame

            fg = binary_fill_holes(fg).astype(np.uint8) * 255
            fg[fg_before_histogram_removal < 255] = 0

            # save_prev_fg
            prev_fg = np.copy(fg)

            # Write to binary movie
            self.out_writer_binary.write(fg)

            # Extract person and write to extracted
            extracted_frame = np.zeros_like(img)
            extracted_frame[fg == 255] = img[fg == 255]

            self.out_writer_extracted.write(extracted_frame)

    def refine_fg_using_kde(self, fg: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Method refines fg map using kde on image pixels
        :param fg: estimated fg, likely to have noise
        :param img: bgr image
        :return: a refined fg
        """
        # Get the img pixels in fg locations
        img_fg_pixels = img[fg == 255]

        # Turn scoring into probability
        log_likelihood_img_pixel_results = self.kde.score_samples(img_fg_pixels)
        log_likelihood_img_pixel_results = np.exp(log_likelihood_img_pixel_results / TEMPERATURE_FACTOR)
        fg_probs = np.zeros_like(fg, dtype=np.float)
        fg_probs[fg == 255] = log_likelihood_img_pixel_results

        # We will build a geodesic distance map using "likely" fg pixels and turn it into confidences
        x, y, w, h = cv2.boundingRect(fg)
        cropped_fg = fg_probs[y:y + h, x:x + w]
        dist_map = GeodisTK.geodesic2d_raster_scan(cropped_fg.astype(np.float32), (cropped_fg > HIGH_PROB_FG_PIXELS_THRESH).astype(np.uint8), 1.0, 2)
        cropped_processed_conf = (dist_map.max() - dist_map)
        cropped_processed_conf[cropped_fg < 0.001] = 0

        # Obtain a 3 level otsu thresholding creating 3 regions
        # When the size of the mid-region is large remove only the small region, else remove both lower ones
        thresholds = threshold_multiotsu(cropped_processed_conf)
        regions = np.digitize(cropped_processed_conf, bins=thresholds)
        reg = 0 if np.sum(regions == 1) > 5000 else 1

        # Create new fg from region thresholding
        cropped_processed_conf[cropped_processed_conf > thresholds[reg]] = 255
        cropped_processed_conf[cropped_processed_conf <= thresholds[reg]] = 0
        fg = np.zeros_like(fg, dtype=np.uint8)
        fg[y:y + h, x:x + w] = cropped_processed_conf.astype(np.uint8)

        # Rerun post process on fg
        fg = self.post_process_fg(fg, opening=False)

        return fg

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sklearn.neighbors import KernelDensity
import GeodisTK
from skimage.filters import threshold_multiotsu
from tqdm import tqdm

from utils import parse_video, get_video_writer


def mask_refinement(fg, do_5_5: bool = True, do_15_15: bool = True) -> np.ndarray:
    """
    Method runs a post process on Bg which keeps only one largest component and runs opening and closing if required
    :param fg: fg mask
    :param do_5_5: signals if to run morphological 5x5 kernel (default True)
    :param do_15_15: signals if to run morphological 15x15 kernel (default True)
    :return: fg after post process
    """
    NOF_COMPONENTS = 1

    # Take only values with 255
    fg[fg < 255] = 0

    # Build connected components, we will take the largest component as it's our fg object
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=4)
    sizes = stats[:, -1]

    # Take the top component
    fg = np.zeros_like(output)
    for i in range(min(NOF_COMPONENTS, nb_components-1)):
        max_label = int(np.argmax(sizes[1:])) + 1
        fg[output == max_label] = 255
        sizes = np.delete(sizes, max_label)

    # Run morphology of opening and closing to clean noise
    if do_5_5:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if do_15_15:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Fix to binary values
    fg[fg < 127] = 0
    fg[fg >= 127] = 255

    return fg

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
                
                _, frame = capture.read()
                self.mog.apply(frame, learningRate=None)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Method applies the mog model to an image
        :param frame: image to apply mog on
        :return: fg mask
        """
        return self.mog.apply(frame, learningRate=0)

class NoiseHandler:
    def __init__(self, capture: cv2.VideoCapture, forward_mog: MixturesOfGaussians, backward_mog: MixturesOfGaussians):
        self.capture = capture
        self.forward_mog = forward_mog
        self.backward_mog = backward_mog
        self.kde = KernelDensity(bandwidth=0.4, kernel='gaussian', atol=0.00000001)

        self.hist_min_val = 0
        self.hist_max_val = 255

    def get_template(self, nof_frames_for_template: int) -> None:
        """
        Method builds a template for kde and histogram
        :param nof_frames_for_template: number of frames to use for template
        :return: None
        """

        # start capture from beginning
        self.capture.set(1, 0)

        self.accumulative_kde_template = np.empty([0, 3])
        self.accumulative_gray_template = np.empty([0, 1])

        # Iterate over num_of_frames_for_template frames and build kde bgr and histogram grayscale template
        for i in range(nof_frames_for_template):
            ret, frame = self.capture.read()
            fg = self.forward_mog.apply(frame)
            fg = mask_refinement(fg)
            frame_template = frame[fg == 255]
            frame_gary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_template_gs = frame_gary[fg == 255]

            # BGR KDE
            self.accumulative_kde_template = np.append(self.accumulative_kde_template, frame_template, axis=0)

            # GS histogram
            self.accumulative_gray_template = np.append(self.accumulative_gray_template, np.expand_dims(frame_template_gs, 1), axis=0)

    def fit_kde(self, nof_frames_for_template: int) -> None:
        """
        Method fits the kde model to the template
        :param nof_frames_for_template: number of frames to use for template
        :return: None
        """
        CDF_HIST_LOW_THRESH = 0.002
        CDF_HIST_HIGH_THRESH = 0.998

        self.get_template(nof_frames_for_template)

        # KDE fitting
        self.kde.fit(self.accumulative_kde_template)

        # GS unused values
        probs, bins = np.histogram(self.accumulative_gray_template.flatten(), bins=30, density=True)
        dx = bins[1] - bins[0]
        cdf = np.cumsum(probs) * dx

        # Remove the extreme values from the histogram
        self.hist_min_val = bins[np.max(np.where(cdf < CDF_HIST_LOW_THRESH))] if len(np.where(cdf < CDF_HIST_LOW_THRESH)[0]) > 0 else 0
        self.hist_max_val = bins[np.min(np.where(cdf > CDF_HIST_HIGH_THRESH))] if len(np.where(cdf > CDF_HIST_HIGH_THRESH)[0]) > 0 else 255

    def refit_kde(self, additional_pixels: np.ndarray) -> None:
        """
        Method refits the kde model to the template
        :param additional_pixels: additional pixels to add to template
        :return: None
        """
        KDE_TEMPLATE_SIZE = 100000
        
        self.accumulative_kde_template = np.append(self.accumulative_kde_template, additional_pixels, axis=0)
        # If len of KDE is above KDE_TEMPLATE_SIZE keep only size
        if len(self.accumulative_kde_template) > KDE_TEMPLATE_SIZE:
            self.accumulative_kde_template = self.accumulative_kde_template[-KDE_TEMPLATE_SIZE:]
        self.kde = KernelDensity(bandwidth=0.4, kernel='gaussian', atol=0.00000001)
        self.kde.fit(self.accumulative_kde_template)

    def score_samples(self, frame: np.ndarray) -> np.ndarray:
        """
        Method scores samples using kde
        :param frame: image to score
        :return: scores
        """

        return self.kde.score_samples(frame)

class BackgroundSubtractor:
    def __init__(self, video_cap_stabilized: cv2.VideoCapture,
                 output_video_path_binary: str, output_video_path_extracted: str):
        
        self.capture = video_cap_stabilized
        self.video_params = parse_video(self.capture)
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.out_writer_binary = get_video_writer(output_video_path_binary, self.video_params, 0)
        self.out_writer_extracted = get_video_writer(output_video_path_extracted, self.video_params)

    def apply(self) -> None:
        """
        Method runs background subtraction on entire video
        :return: None
        """
        self.forward_mog = MixturesOfGaussians(num_frames=self.num_frames, num_mixtures=5, var_threshold=4, is_forward=True)
        self.backward_mog = MixturesOfGaussians(num_frames=self.num_frames, num_mixtures=5, var_threshold=4, is_forward=False)
        self.forward_mog.fit(self.capture, 5)
        self.backward_mog.fit(self.capture, 5)

        nof_frames_for_template = 5
        self.noiseHandler = NoiseHandler(self.capture, self.forward_mog, self.backward_mog)
        self.noiseHandler.fit_kde(nof_frames_for_template)
        self.last_trained_frame = nof_frames_for_template # We start with 5 frames for template
        self.hist_min_val = self.noiseHandler.hist_min_val
        self.hist_max_val = self.noiseHandler.hist_max_val

        self.run_substraction()

        self.out_writer_binary.release()
        self.out_writer_extracted.release()

    def run_substraction(self) -> None:
        """
        Runs background subtraction on video frame by frame
        :return: None
        """

        # start capture from beginning
        self.capture.set(1, 0)

        # prev_fg is used to calculate the difference from prev frame used to "signal" problematic frames
        prev_fg = None

        # Iterate over all frames and run mog
        for frame_id in tqdm(range(self.num_frames)):
            _, frame = self.capture.read()

            # Use forward mog for first half of frames and backward for second half
            mog = self.forward_mog if frame_id < self.num_frames//2 else self.backward_mog
            fg = mog.apply(frame)

            # Refine mask
            fg = mask_refinement(fg)
            inital_background = fg < 255

            # Remove the unused values according to hist
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg[np.logical_or(frame_gray < self.hist_min_val, frame_gray > self.hist_max_val)] = 0

            # the first 5 were used for templating, so we skip them
            if frame_id > 5:
                # Get the time diff of fg in the direction of the movement
                seg_time_diff = fg - prev_fg
                seg_time_diff = seg_time_diff * (seg_time_diff > 0)
                # If diff is large, use kde refinement to clear noise
                if (np.sum(seg_time_diff) / 255) > 7500:
                    fg = self.kde_mask_refinement(fg, frame)

                # If change is small and kde hasn't been retrained in last 5 frames, retrain kde
                elif ((np.sum(seg_time_diff) / 255) < 6500) and (frame_id - self.last_trained_frame >= 5):
                    self.noiseHandler.refit_kde(frame[fg > 0])
                    self.last_trained_frame = frame_id

            fg = binary_fill_holes(fg).astype(np.uint8) * 255
            fg[inital_background] = 0

            # save_prev_fg
            prev_fg = np.copy(fg)

            # Write to binary movie
            self.out_writer_binary.write(fg)

            # Extract person and write to extracted
            extracted_frame = np.zeros_like(frame)
            extracted_frame[fg == 255] = frame[fg == 255]

            self.out_writer_extracted.write(extracted_frame)

    def kde_mask_refinement(self, fg: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Method refines fg map using kde on image pixels
        :param fg: estimated fg, likely to have noise
        :param frame: bgr image
        :return: a refined fg
        """
        HIGH_PROB_FG_PIXELS_THRESH = 0.8
        TEMPERATURE_FACTOR = 48

        # Turn scoring into probability
        log_likelihood_frame_pixel_results = self.noiseHandler.score_samples(frame[fg == 255])
        log_likelihood_frame_pixel_results = np.exp(log_likelihood_frame_pixel_results / TEMPERATURE_FACTOR)
        fg_probs = np.zeros_like(fg, dtype=np.float)
        fg_probs[fg == 255] = log_likelihood_frame_pixel_results

        # We will build a geodesic distance map using "likely" fg pixels and turn it into confidences
        x, y, w, h = cv2.boundingRect(fg)
        cropped_fg = fg_probs[y:y+h, x:x+w]
        dist_map = GeodisTK.geodesic2d_raster_scan(cropped_fg.astype(np.float32), (cropped_fg > HIGH_PROB_FG_PIXELS_THRESH).astype(np.uint8), 1.0, 2)
        cropped_processed_conf = (dist_map.max() - dist_map)
        cropped_processed_conf[cropped_fg < 0.002] = 0

        # Obtain a 3 level otsu thresholding creating 3 regions
        # When the size of the mid-region is large remove only the small region, else remove both lower ones
        thresholds = threshold_multiotsu(cropped_processed_conf)
        regions = np.digitize(cropped_processed_conf, bins=thresholds)
        reg = 0 if np.sum(regions == 1) > 5000 else 1

        # Create new fg from region thresholding
        cropped_processed_conf[cropped_processed_conf > thresholds[reg]] = 255
        cropped_processed_conf[cropped_processed_conf <= thresholds[reg]] = 0
        fg = np.zeros_like(fg, dtype=np.uint8)
        fg[y:y+h, x:x+w] = cropped_processed_conf.astype(np.uint8)

        # Rerun post process on fg
        return mask_refinement(fg, do_15_15=False)
    
def substact_background(video_cap_stabilized: cv2.VideoCapture, output_video_path_binary: str, output_video_path_extracted: str) -> None:
    backgroundSubtractor = BackgroundSubtractor(video_cap_stabilized, output_video_path_binary, output_video_path_extracted)
    backgroundSubtractor.apply()

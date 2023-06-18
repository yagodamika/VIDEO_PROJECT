import cv2
import numpy as np
from tqdm import tqdm

from utils.video_utils import get_video_parameters, build_out_writer


class VideoStabilizer:
    def __init__(self, video_cap: cv2.VideoCapture, output_video_path: str):
        self.cap = video_cap
        self.first_frame = None
        self.orig_frame_shape = None

        self.video_params = get_video_parameters(self.cap)
        self.u, self.v = 0, 0
        self.transform_matrix = np.eye(3)

        self.out_writer = build_out_writer(output_video_path, self.video_params)

        self.extract_and_write_first_frame()
        self.stabilize_and_write_frames()
        self.out_writer.release()

    def extract_and_write_first_frame(self) -> None:
        """
        Method extracts and writes the first frame to the stabilized video
        :return: None
        """
        ret, frame = self.cap.read()
        self.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.out_writer.write(frame)

    def stabilize_and_write_frames(self) -> None:
        """
        Method iterates over all frames, stabilizes each frame to the first frame one using feature matching
        :return: None
        """
        print("Video Stabilization phase")
        for i in tqdm(range(1, self.video_params['frame_count'])):
            ret, next_frame = self.cap.read()

            # Get transformation matrix to the previous frame
            transformation = self.find_single_ts_transformation_matrix_using_feature_matching(next_frame)

            # Warp current image and write it to stabilized output video
            frame_stabilized = cv2.warpPerspective(next_frame, transformation, (self.video_params["width"], self.video_params["height"]))

            self.out_writer.write(frame_stabilized)

    def find_single_ts_transformation_matrix_using_feature_matching(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculates transformation from a frame to the first frame
        :param frame: the current frame
        :return: transformation matrix
        """
        # Sift for extracting keypoints and computing descriptors
        sift = cv2.SIFT_create()

        kp1, descriptors_1 = sift.detectAndCompute(self.first_frame, None)
        kp2, descriptors_2 = sift.detectAndCompute(frame, None)

        # feature matching using brute force matcher
        bf = cv2.BFMatcher()
        matches = bf.match(descriptors_1, descriptors_2)

        src_pts = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

        # Find homography using points from matches
        transformation, outliers = cv2.findHomography(dst_pts, src_pts, method=cv2.RANSAC, confidence=0.99)

        return transformation

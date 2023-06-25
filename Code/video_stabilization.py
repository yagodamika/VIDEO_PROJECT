import cv2
import numpy as np
from tqdm import tqdm
from utils.video_utils_ref import get_video_parameters, build_out_writer


class Stabilizer:
    def __init__(self, capture: cv2.VideoCapture, output_path: str):
        self.cap = capture
        self.video_parameters = get_video_parameters(self.cap)
        self.out_writer = build_out_writer(output_path, self.video_parameters)
        self.transform_matrix = np.eye(3)
        
        # get first frame and write it to output video
        _, frame = self.cap.read()
        self.out_writer.write(frame)
        self.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.apply_stabilization()
        self.out_writer.release()


    def apply_stabilization(self) -> None:
        """
        Method iterates over all frames, stabilizes each frame to the first frame one using feature matching
        :return: None
        """
        print("Video Stabilization")
        for i in tqdm(range(1, self.video_parameters['frame_count'])):
            _, frame = self.cap.read()
            # calculate transformation to the frame before
            transformation_matrix = self.get_transformation_mat(frame)
            # warp frame and write it to output video
            frame_stabilized = cv2.warpPerspective(frame, transformation_matrix, (self.video_parameters["width"], self.video_parameters["height"]))
            self.out_writer.write(frame_stabilized)

    def get_transformation_mat(self, frame: np.ndarray) -> np.ndarray:
        """
        Get transformation matrix from a frame to the first frame
        
        Args:
        frame: matted image

        Return:
        trans_mat: the transformation matrix
        """
        sift = cv2.SIFT_create()
        key_points_first, descriptors_first = sift.detectAndCompute(self.first_frame, None)
        key_points, descriptors = sift.detectAndCompute(frame, None)
        # perform feature matching 
        brute_force_matcher = cv2.BFMatcher()
        matches = brute_force_matcher.match(descriptors_first, descriptors)

        source_points = [key_points_first[match.queryIdx].pt for match in matches]
        dest_points = [key_points[match.trainIdx].pt for match in matches]
        np_source_points = np.float32(source_points)
        np_dest_points = np.float32(dest_points)
        
        # get homography using matching points
        trans_mat, outliers = cv2.findHomography(np_dest_points.reshape(-1, 1, 2), np_source_points.reshape(-1, 1, 2),
                                                      method=cv2.RANSAC, confidence=0.99)

        return trans_mat

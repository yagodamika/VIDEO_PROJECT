import argparse
import cv2
import os

from video_stabilization import stablize_video
from background_subtraction import substact_background
from matting import paste_object_on_background
from utils import Timing

ID1 = '313514044'
ID2 = '208687129'

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--input_video', type=str, default="./Inputs/INPUT.mp4", help='Path to input video file')
    parser.add_argument('--background_img', type=str, default="./Inputs/background.jpg", help='Path to input background file')
    parser.add_argument('--output_video_folder', type=str, default="./Outputs", help='Path to output video folder')
    args = parser.parse_args()

    # Setup paths
    os.makedirs(args.output_video_folder, exist_ok=True)
    timing_path = os.path.join(args.output_video_folder, "timing.json")
    tracking_path = os.path.join(args.output_video_folder, "tracking.json")
    stabilize_path = os.path.join(args.output_video_folder, f"stabilize_{ID1}_{ID2}.avi")
    extracted_path = os.path.join(args.output_video_folder, f"extracted_{ID1}_{ID2}.avi")
    binary_path = os.path.join(args.output_video_folder, f"binary_{ID1}_{ID2}.avi")
    matted_path = os.path.join(args.output_video_folder, f"matted_{ID1}_{ID2}.avi")
    alpha_path = os.path.join(args.output_video_folder, f"alpha_{ID1}_{ID2}.avi")
    output_path = os.path.join(args.output_video_folder, f"OUTPUT_{ID1}_{ID2}.avi")

    # Load input video
    timing = Timing(timing_path)
    video_cap = cv2.VideoCapture(args.input_video)

    # Run video stabilization
    print("Video Stabilization")
    stablize_video(video_cap, stabilize_path)
    timing.log_time("time_to_stabilize")

    # Run background subtraction
    print("Background Subtraction")
    video_cap_stabilized = cv2.VideoCapture(stabilize_path)
    substact_background(video_cap_stabilized, binary_path, extracted_path)
    timing.log_time("time_to_binary")

    # Run matting and detection
    print("Matting and Detection")
    video_cap_stabilized = cv2.VideoCapture(stabilize_path)
    video_cap_binary = cv2.VideoCapture(binary_path)
    paste_object_on_background(video_cap_stabilized, video_cap_binary, args.background_img, alpha_path, matted_path, output_path, tracking_path)

    timing.log_time("time_to_alpha")
    timing.log_time("time_to_matted")
    timing.log_time("time_to_output")
    timing.dump()

if __name__ == '__main__':
    main()

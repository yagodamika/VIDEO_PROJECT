import os

import cv2

from background_subtraction import BackgroundSubtractor
from matting import MattingMaker
from video_stabilization import Stabilizer
from utils.general_utils_ref import Timing

INPUT_VIDEO_PATH = "./Inputs/INPUT.mp4"
BACKGROUND_PATH = "./Inputs/background.jpg"
ID1 = '300508850'
ID2 = '021681283'

OUTPUT_FOLDER = "./Outputs"
STABILIZE = f"stabilize_{ID1}_{ID2}.avi"
EXTRACTED = f"extracted_{ID1}_{ID2}.avi"
BINARY = f"binary_{ID1}_{ID2}.avi"
MATTED = f"matted_{ID1}_{ID2}.avi"
ALPHA = f"alpha_{ID1}_{ID2}.avi"
OUTPUT = f"OUTPUT_{ID1}_{ID2}.avi"
TIMING = "timing.json"
TRACKING = "tracking.json"


def main():
    # Load input video
    timing_object = Timing(os.path.join(OUTPUT_FOLDER, TIMING))
    video_cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    # # Run video stabilization
    Stabilizer(video_cap, os.path.join(OUTPUT_FOLDER, STABILIZE))
    timing_object.write_time_of_stage("time_to_stabilize")

    # Run background subtraction
    video_cap_stabilized = cv2.VideoCapture(os.path.join(OUTPUT_FOLDER, STABILIZE))
    BackgroundSubtractor(video_cap_stabilized, os.path.join(OUTPUT_FOLDER, BINARY), os.path.join(OUTPUT_FOLDER, EXTRACTED))
    timing_object.write_time_of_stage("time_to_binary")

    # Run matting and detection
    video_cap_stabilized = cv2.VideoCapture(os.path.join(OUTPUT_FOLDER, STABILIZE))
    video_cap_binary = cv2.VideoCapture(os.path.join(OUTPUT_FOLDER, BINARY))
    output_path_alpha = os.path.join(OUTPUT_FOLDER, ALPHA)
    output_path_matting = os.path.join(OUTPUT_FOLDER, MATTED)
    output_path_final = os.path.join(OUTPUT_FOLDER, OUTPUT)
    output_detections_json_path = os.path.join(OUTPUT_FOLDER, TRACKING)
    MattingMaker(video_cap_stabilized, video_cap_binary, BACKGROUND_PATH,
                          output_path_alpha, output_path_matting, output_path_final, output_detections_json_path)

    # Note as all 3 happen together, we finished writing them at the same time
    timing_object.write_time_of_stage("time_to_alpha")
    timing_object.write_time_of_stage("time_to_matted")
    timing_object.write_time_of_stage("time_to_output")

    timing_object.write_timing_to_json()


if __name__ == '__main__':
    main()

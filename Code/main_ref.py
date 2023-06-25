import cv2
import os

from video_stabilization import Stabilizer
from background_subtraction import substactBackground
from matting import MattingMaker
from utils import Timing

INPUT_VIDEO_PATH = "./Inputs/INPUT.mp4"
BACKGROUND_PATH = "./Inputs/background.jpg"
ID1 = '313514044'
ID2 = '208687129'

OUTPUT_FOLDER = "./Outputs"
STABILIZE_PATH = os.path.join(OUTPUT_FOLDER, f"stabilize_{ID1}_{ID2}.avi")
EXTRACTED_PATH = os.path.join(OUTPUT_FOLDER, f"extracted_{ID1}_{ID2}.avi")
BINARY_PATH = os.path.join(OUTPUT_FOLDER, f"binary_{ID1}_{ID2}.avi")
MATTED_PATH = os.path.join(OUTPUT_FOLDER, f"matted_{ID1}_{ID2}.avi")
ALPHA_PATH = os.path.join(OUTPUT_FOLDER, f"alpha_{ID1}_{ID2}.avi")
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"OUTPUT_{ID1}_{ID2}.avi")
TIMING_PATH = os.path.join(OUTPUT_FOLDER, "timing.json")
TRACKING_PATH = os.path.join(OUTPUT_FOLDER, "tracking.json")


def main():
    # Load input video
    timing = Timing(TIMING_PATH)
    video_cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    # Run video stabilization
    Stabilizer(video_cap, STABILIZE_PATH)
    timing.log_time("time_to_stabilize")

    # Run background subtraction
    video_cap_stabilized = cv2.VideoCapture(STABILIZE_PATH)
    substactBackground(video_cap_stabilized, BINARY_PATH, EXTRACTED_PATH)
    timing.log_time("time_to_binary")

    # Run matting and detection
    video_cap_stabilized = cv2.VideoCapture(STABILIZE_PATH)
    video_cap_binary = cv2.VideoCapture(BINARY_PATH)
    MattingMaker(video_cap_stabilized, video_cap_binary, BACKGROUND_PATH,
                          ALPHA_PATH, MATTED_PATH, OUTPUT_PATH, TRACKING_PATH)

    # Note as all 3 happen together, we finished writing them at the same time
    timing.log_time("time_to_alpha")
    timing.log_time("time_to_matted")
    timing.log_time("time_to_output")

    timing.dump()


if __name__ == '__main__':
    main()

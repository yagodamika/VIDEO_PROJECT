from dataclasses import dataclass
import cv2
import json
import time


@dataclass
class params:
    """Data class holding parameters for logging."""
    fourcc: int
    fps: int
    frame_height: int
    frame_width: int
    frame_num: str

def parse_video(capture: cv2.VideoCapture) -> params:
    """
    Extracts parameters from an OpenCV capture object representing a video.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: params. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_params = params(fourcc=fourcc, fps=fps,
                          frame_height=height, frame_width=width, frame_num=frame_num)
    
    return video_params


def get_video_writer(output_path: str, video_parameters: params, mode=None):
    """
    This function created a video writer using given video parameteres
    Args:
        output_path: path for the output video writer.
        video_parameters: video parameters.
    Returns:
       writer: 

    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = video_parameters.frame_width
    height = video_parameters.frame_height
    writer = cv2.VideoWriter(output_path, fourcc, video_parameters.fps, (width, height), mode)
    return writer

class Timing:
    def __init__(self, output_path: str):
        self.start_time = time.time()
        self.timing_dict = {
                "time_to_stabilize": 0,
                "time_to_binary": 0,
                "time_to_alpha": 0,
                "time_to_matted": 0,
                "time_to_output": 0
            }
        self.output_path = output_path

    def log_time(self, stage: str) -> None:
        self.timing_dict[stage] = time.time() - self.start_time

    def dump(self) -> None:
        with open(self.output_path, 'w') as json_handler:
            json.dump(self.timing_dict, json_handler, indent=4)

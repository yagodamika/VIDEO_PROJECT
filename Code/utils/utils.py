from dataclasses import dataclass
import cv2


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

import json
import time

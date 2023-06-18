from typing import Dict

import cv2


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width, "frame_count": frame_count}


def build_out_writer(output_video_path: str, video_params: Dict[str, any], mode=None):
    """
    Function builds an output video writer according to video params
    :param output_video_path:
    :param video_params:
    :param mode:
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    new_width = video_params['width']
    new_height = video_params['height']
    out_writer = cv2.VideoWriter(output_video_path, fourcc, video_params['fps'], (new_width, new_height), mode)
    return out_writer

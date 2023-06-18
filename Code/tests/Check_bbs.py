
import cv2
import json
from matplotlib import pyplot as plt

test_video_path = "../Outputs/matted_300508850_021681283.avi"
cap = cv2.VideoCapture(test_video_path)

bb_json_path = "..//Outputs/tracking.json"
bbs_test = json.load(open(bb_json_path, "r"))
frame_num = 1
while True:
    ret, frame = cap.read()
    if not(ret):
        break
    bb = bbs_test[str(frame_num)]
    row_center = bb[0]
    col_center = bb[1]
    half_height = bb[2]
    half_width = bb[3]
    print(f"row center: {row_center} \ncol center: {col_center} \nhalf height {half_height} \nhalf width {half_width}")
    plt.imshow(frame)
    plt.show()

    frame_num +=1

"""
This script to create .csv videos frames action annotation file.

- It will play a video frame by frame control the flow by [a] and [d]
 to play previos or next frame.
- Open the annot_file (.csv) and label each frame of video with number
 of action class.
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class_names = ['sitting','throwing','shooting','fall down']  #WHEN FINISH THE ANOTATION FOR CHECK BY VIDEO.

video_folder = '../Data/falldata/Home/videos'
annot_file = '../Data/Home_new.csv'

# index_video_to_play = 1  # Choose video to play.


def create_csv(folder):
    list_file = sorted(os.listdir(folder))
    cols = ['video', 'frame', 'label']
    df = pd.DataFrame(columns=cols)
    for fil in list_file:
        cap = cv2.VideoCapture(os.path.join(folder, fil))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video = np.array([fil] * frames_count)
        frame = np.arange(1, frames_count + 1)
        label = np.array([0] * frames_count)
        rows = np.stack([video, frame, label], axis=1)
        df = df.append(pd.DataFrame(rows, columns=cols),
                       ignore_index=True)
        cap.release()
    df.to_csv(annot_file, index=False)


if not os.path.exists(annot_file):
    create_csv(video_folder)
list_file = sorted(os.listdir(video_folder))

for index_video_to_play in range(0,len(list_file)):
    annot = pd.read_csv(annot_file)
    video_list = annot.iloc[:, 0].unique()
    video_file = os.path.join(video_folder, video_list[index_video_to_play])
    print(os.path.basename(video_file))

    annot = annot[annot['video'] == video_list[index_video_to_play]].reset_index(drop=True)
    frames_idx = annot.iloc[:, 1].tolist()

    cap = cv2.VideoCapture(video_file)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert frames_count == len(frames_idx), 'frame count not equal! {} and {}'.format(
        len(frames_idx), frames_count
    )

    i = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cls_name = class_names[int(annot.iloc[i, -1]) - 1]
            # frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
            frame = cv2.putText(frame, 'Frame: {} Pose: {}'.format(i+1, cls_name),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.resizeWindow("frame",1280,640);
            cv2.imshow(video_list[index_video_to_play], frame)
            # cv2.imshow('frame', frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('d') or key == ord('D'):
                i += 1
                continue
            elif key == ord('a') or key == ord('A'):
                i -= 1
                continue
        else:
            break

cap.release()
cv2.destroyAllWindows()

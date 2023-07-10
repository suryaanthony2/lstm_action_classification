import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp
import matplotlib.pyplot as plt
from time import time
import cv2
import numpy as np
import json
import pathlib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lib import prediction

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils 
    
with open(os.fspath(pathlib.Path(__file__).parent.parent / "config.json"), "r") as f:
    cfg = json.load(f)

padding = cfg["padding"]

def detectPose(image, pose, display=True):  

    output_image = image.copy()
    
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(imageRGB)
    
    height, width, _ = image.shape
    
    landmarks = []
    
    landmarks_norm = []
    
    if results.pose_landmarks:
    
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i < 11 and i != 0:
                continue
            landmarks_norm.append((landmark.x, landmark.y, landmark.z))
            
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    if display:
    
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        
        return output_image, landmarks, landmarks_norm
    

def make_dataset(classes, path, quiet=False, show=False):
    temp_x = []
    temp_y = []
    temp_z = []
    target = []

    for i, cls in enumerate(classes): 
        file_path = os.listdir(path + "\\" + cls)

        for file_name in file_path:
            coord_list_norm = show_video(path + "\\" + cls + "\\" + file_name, show=show)
            
            if quiet:
                pass
            else:
                print(file_name + " sudah selesai")
            
            coord_list_norm = [x for x in coord_list_norm if x]
            
            train_input = np.array(coord_list_norm)
            
            seq_x, seq_y, seq_z = pad_array(train_input, padding)
            
            temp_x.extend(seq_x)
            temp_y.extend(seq_y)
            temp_z.extend(seq_z)
            
            target.append(i)
    
    temp_x = np.array(temp_x)
    temp_y = np.array(temp_y)
    temp_z = np.array(temp_z)

    x = np.vstack((temp_x, temp_y, temp_z))

    y = np.array(target)

    return x, y

def get_coordinates(path):
    coord_list_norm = show_video(path, show=True)
    
    coord_list_norm = [x for x in coord_list_norm if x]
    
    train_input = np.array(coord_list_norm)
        
    seq_x, seq_y, seq_z = pad_array(train_input, padding)

    x = np.vstack((seq_x, seq_y, seq_z))
    
    return x


def detect_real_time(camera, model, flip=False):
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=1)

    min_fps = 100
    max_fps = 0
    delay_frame = 0
    maxlen = padding

    x = np.full((1, maxlen, 23), 2.0)
    y = np.full((1, maxlen, 23), 2.0)
    z = np.full((1, maxlen, 23), 2.0)
    
    count = 0
    delay = 4
    
    video = cv2.VideoCapture(camera)

    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

    video.set(3,640)
    video.set(4,360)

    time1 = 0
    while video.isOpened():
    
        ok, frame = video.read()
    
        if not ok:

            break

        if flip:
            frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ =  frame.shape

        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, landmarks, landmarks_norm = detectPose(frame, pose_video, display=False)
        
        landmarks_norm = [x for x in landmarks_norm if x]
        
        train_input = np.array(landmarks_norm)
        
        if train_input.size == 0:
            cv2.imshow('Pose Detection', frame)
    
            k = cv2.waitKey(1) & 0xFF

            if(k == 27):
                break
                
            if cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:        
                break
            continue
        
        if count < maxlen:
            x[:, count, :] = train_input[:, 0]
            y[:, count, :] = train_input[:, 1]
            z[:, count, :] = train_input[:, 2]
            count += 1
        else:
            x = np.roll(x, -1, axis=1)
            x[:, -1, :] = train_input[:, 0]
            y = np.roll(y, -1, axis=1)
            y[:, -1, :] = train_input[:, 1]
            
            z = np.roll(z, -1, axis=1)
            z[:, -1, :] = train_input[:, 2]
            
        if delay == 4:
            move, prediction_prob = prediction.classify(x, y, z, model)
            delay = 0
        else :
            print(min_fps)
            delay += 1
        
        time2 = time()
    
        if (time2 - time1) > 0:

            fps = int(1.0 / (time2 - time1))
            if delay_frame == 10:
                if fps < min_fps:
                    min_fps = fps
            else:
                delay_frame += 1
            if fps > max_fps:
                max_fps = fps

            cv2.putText(frame, 'FPS: {}'.format(fps), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.putText(frame, 'prediction: {}'.format(prediction_prob), (10, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.putText(frame, 'move: ' + move, (10, 150),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
        time1 = time2
        cv2.imshow('Pose Detection', frame)

        k = cv2.waitKey(1) & 0xFF

        if(k == 27):

            break
            
        if cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:        
            break
            
    video.release()

    cv2.destroyAllWindows()

    print(f"Min fps = {min_fps}, max fps = {max_fps}")

def get_frame_dist():
    print("Menghitung jumlah frame, mohon tunggu!")  
    frame_in_video = {}
    directory = ["train", "test"]
    path = str(pathlib.Path().resolve()) + "\\video\\"

    for folder in directory:    
        classes = os.listdir(path + folder)
        for i, cls in enumerate(classes): 
            file_path = os.listdir(path + folder + "\\" + cls)
            for file_name in file_path:
                coord_list_norm = show_video(path + folder + "\\" + cls + "\\" + file_name)

                coord_list_norm = [x for x in coord_list_norm if x]

                if len(coord_list_norm) not in frame_in_video:
                    frame_in_video[len(coord_list_norm)] = 1
                else:
                    frame_in_video[len(coord_list_norm)] += 1  
    
    d = dict(sorted(frame_in_video.items()))
    return d

def pad_array(train_input, padding):
    seq_x = train_input[:,:,0]
    seq_y = train_input[:,:,1]
    seq_z = train_input[:,:,2]
    
    seq_x = pad_sequences([seq_x], maxlen=padding, dtype='float32', padding='post', value=2)
    seq_y = pad_sequences([seq_y], maxlen=padding, dtype='float32', padding='post', value=2)
    seq_z = pad_sequences([seq_z], maxlen=padding, dtype='float32', padding='post', value=2)

    return seq_x, seq_y, seq_z

def show_video(path, flip=False, show=False):
    coord_list_norm = []

    if show:
        cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

    maxlen = 120

    video = cv2.VideoCapture(path)

    video.set(3,1280)
    video.set(4,720)

    time1 = 0

    while video.isOpened():

        ok, frame = video.read()

        if not ok:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ =  frame.shape

        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, _, landmarks_norm = detectPose(frame, pose, display=False)

        coord_list_norm.append(landmarks_norm)

        time2 = time()

        if (time2 - time1) > 0:

            frames_per_second = 1.0 / (time2 - time1)

            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
        time1 = time2

        if show:
            cv2.imshow('Pose Detection', frame)

        k = cv2.waitKey(1) & 0xFF

        if(k == 27):
            break

    video.release()

    cv2.destroyAllWindows()

    return coord_list_norm
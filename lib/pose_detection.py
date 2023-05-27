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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils 
    
with open(os.fspath(pathlib.Path(__file__).parent.parent / "config.json"), "r") as f:
    cfg = json.load(f)

maxlen = cfg["maxlen"]

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    landmarks_norm = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i < 11 and i != 0:
                continue
            landmarks_norm.append((landmark.x, landmark.y, landmark.z))
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks, landmarks_norm
    

def get_coordinates(classes, path, quiet=False, show=False):
    temp_x = []
    temp_y = []
    temp_z = []
    target = []
    # Create named window for resizing purposes
    #cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

    for i, cls in enumerate(classes): 
        coord_list_norm = []
        file_path = os.listdir(path + "\\" + cls)
        # Initialize the VideoCapture object to read from the webcam.
        # video = cv2.VideoCapture(1)

        # Initialize the VideoCapture object to read from a video stored in the disk.
        for file_name in file_path:
            video = cv2.VideoCapture(path + "\\" + cls + "\\" + file_name)

            # Set video camera size
            video.set(3,1280)
            video.set(4,720)

            # Initialize a variable to store the time of the previous frame.
            time1 = 0
            # Iterate until the video is accessed successfully.
            while video.isOpened():
        
                # Read a frame.
                ok, frame = video.read()
        
                # Check if frame is not read properly.
                if not ok:
            
                    # Break the loop.
                    break
        
                # Flip the frame horizontally for natural (selfie-view) visualization.
                #frame = cv2.flip(frame, 1)
        
                # Get the width and height of the frame
                frame_height, frame_width, _ =  frame.shape
        
                # Resize the frame while keeping the aspect ratio.
                frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
                # Perform Pose landmark detection.
                frame, landmarks, landmarks_norm = detectPose(frame, pose, display=False)
        
                coord_list_norm.append(landmarks_norm)
        
                # Set the time for this frame to the current time.
                time2 = time()
        
                # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
                if (time2 - time1) > 0:
        
                    # Calculate the number of frames per second.
                    frames_per_second = 1.0 / (time2 - time1)
            
                    # Write the calculated number of frames per second on the frame. 
                    cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
                # Update the previous frame time to this frame time.
                # As this frame will become previous frame in next iteration.
                time1 = time2 
        
                # Display the frame.
                if show:
                    cv2.imshow('Pose Detection', frame)
        
        
                # Wait until a key is pressed.
                # Retreive the ASCII code of the key pressed
                    k = cv2.waitKey(1) & 0xFF
        
                # Check if 'ESC' is pressed.
                    if(k == 27):
            
                    # Break the loop.
                        break

            # Release the VideoCapture object.
            video.release()

            # Close the windows.
            cv2.destroyAllWindows()
            
            if quiet:
                pass
            else:
                print(file_name + " sudah selesai")
            
            coord_list_norm = [x for x in coord_list_norm if x]
            
            train_input = np.array(coord_list_norm)
            
            seq_x = train_input[:,:,0]
            seq_y = train_input[:,:,1]
            seq_z = train_input[:,:,2]
            
            seq_x = pad_sequences([seq_x], maxlen=maxlen, dtype='float32', padding='post', value=2)
            seq_y = pad_sequences([seq_y], maxlen=maxlen, dtype='float32', padding='post', value=2)
            seq_z = pad_sequences([seq_z], maxlen=maxlen, dtype='float32', padding='post', value=2)
            
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

def get_coordinates_lstm(path):
    coord_list_norm = []
    # Initialize the VideoCapture object to read from the webcam.
    # video = cv2.VideoCapture(1)

    # Create named window for resizing purposes
    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

    maxlen = 120
    # Initialize the VideoCapture object to read from a video stored in the disk.
    video = cv2.VideoCapture(path)

    # Set video camera size
    video.set(3,1280)
    video.set(4,720)

    # Initialize a variable to store the time of the previous frame.
    time1 = 0
    # Iterate until the video is accessed successfully.
    while video.isOpened():
    
        # Read a frame.
        ok, frame = video.read()
    
        # Check if frame is not read properly.
        if not ok:
        
            # Break the loop.
            break
    
        # Flip the frame horizontally for natural (selfie-view) visualization.
        #frame = cv2.flip(frame, 1)
    
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
    
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
        # Perform Pose landmark detection.
        frame, landmarks, landmarks_norm = detectPose(frame, pose, display=False)

        coord_list_norm.append(landmarks_norm)
    
        # Set the time for this frame to the current time.
        time2 = time()
    
        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:
    
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)
        
            # Write the calculated number of frames per second on the frame. 
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        time1 = time2
    
        # Display the frame.
        cv2.imshow('Pose Detection', frame)
    
    
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF
    
        # Check if 'ESC' is pressed.
        if(k == 27):
        
            # Break the loop.
            break

    # Release the VideoCapture object.
    video.release()

    # Close the windows.
    cv2.destroyAllWindows()
    
    coord_list_norm = [x for x in coord_list_norm if x]
    
    train_input = np.array(coord_list_norm)
        
    seq_x = train_input[:,:,0]
    seq_y = train_input[:,:,1]
    seq_z = train_input[:,:,2]
        
    seq_x = pad_sequences([seq_x], maxlen=maxlen, dtype='float32', padding='post', value=2)
    seq_y = pad_sequences([seq_y], maxlen=maxlen, dtype='float32', padding='post', value=2)
    seq_z = pad_sequences([seq_z], maxlen=maxlen, dtype='float32', padding='post', value=2)

    x = np.vstack((seq_x, seq_y, seq_z))
    
    return x




#mungkin pas buat dataset bisa tidak ditampilkan videonya dengan komen di bagian imshow
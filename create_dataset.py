import os
import pickle # save result
import mediapipe as mp # landmark detector
import cv2

# Mediapipe Hand Detection
mp_hands = mp.solutions.hands # access hand detection modul
mp_drawing = mp.solutions.drawing_utils # applying landmark
mp_drawing_styles = mp.solutions.drawing_styles # styling landmark

# detect hands as a static image (not a video)
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# load dataset
DATA_DIR = './data'

# make an array to save landmark for each image and its label
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    # read every image from each class
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [] # save coordinate x and y from every image

        # read each image with opencv
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))

        # convert to RGB since Mediapipe used BGR format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image with coordinate dot if detected
        results = hands.process(img_rgb) 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x) # horizontal
                    data_aux.append(y) # vertical
            
            # add feature and label from data_aux to data and labels
            data.append(data_aux)
            labels.append(dir_)

# save extracted data (data and label) to file pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
import pickle
import cv2
import mediapipe as mp
import numpy as np # processing numeric array data

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model'] 

# access camera
cap = cv2.VideoCapture(0)

# Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Class label
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 
    3: 'D', 4: 'E', 5: 'F', 
    6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V',
    21: 'W', 22: 'X', 23: 'Y'
}
    
# real time prediction
while True:
    # read camera frame
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break
    
    # convert to RGB
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # if hands detected at least 1 hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # draw landmark for detected hand
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # save coordinate landmark
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # if only 1 hand detected, do 42 [0] padding so the size is 84
        if len(results.multi_hand_landmarks) == 1:
            data_aux.extend([0] * 42)  

        # predict label based on gesture
        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    # show frame
    cv2.imshow('frame', frame)

    # exit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean resource
cap.release()
cv2.destroyAllWindows()

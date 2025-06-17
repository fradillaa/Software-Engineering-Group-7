import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style = 'color:orange;'>Selamat Datang di Website Deteksi Bahasa Isyarat ^^</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption('Silahkan aktifkan webcam untuk memulai deteksi gerakan tangan.')
run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        st.write("Camera error!")
        st.write("Silahkan Nyalakan Kamera Anda.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        if len(results.multi_hand_landmarks) == 1:
            data_aux.extend([0] * 42)

        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            cv2.putText(frame, predicted_character, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
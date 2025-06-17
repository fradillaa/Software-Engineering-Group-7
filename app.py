import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import random
import time
import base64
import streamlit as st
import base64

def set_bg(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg("images/background.jpg")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "page_history" not in st.session_state:
    st.session_state.page_history = []

# Navigasi dengan tombol
def go_to(page_name):
    if "page_history" not in st.session_state:
        st.session_state.page_history = []
    st.session_state.page_history.append(st.session_state.page)
    st.session_state.page = page_name

# Function button back
def go_back_menu():
    if st.session_state.page_history:
        st.session_state.page = st.session_state.page_history.pop()
    else:
        st.session_state.page = "home"

# Halaman 1: Home
if st.session_state.page == "home":
    st.markdown(
    """
    <style>
        .centered {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        h1 {
            color: #A3B18A !important;
            font-size: 60px !important;
            margin-bottom: 10px;
            text-shadow: 
                -1px -1px 0 #f6eedf,  
                1px -1px 0 #f6eedf,
                -1px 1px 0 #f6eedf,
                1px 1px 0 #ffffff;
        }

        p {
            font-size: 20px !important;
            color: #f4e8da;
            width: 100%;
            max-width: 900px;
            text-align: center;
            margin-bottom: 10px;
            background-color: rgba(0, 0, 0, 0.2); /* transparan */
            padding: 20px;
            border-radius: 15px;
        }

        .stButton > button {
            background-color: #A3B18A;
            color: #ffffff;
            padding: 10px 10px;
            font-size: 15px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            white-space: nowrap; 
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }

        .stButton > button:hover {
            background-color: #8f9d72;
            transform: scale(1.03);
        }
    </style>

    <div class="centered">
        <h1><b>MIMICO</b></h1>
        <p>
        <b>Welcome to MIMICO!<br>
        Learning sign language becomes easy, engaging, and totally beginner-friendly. Whether you're learning for fun, for friends, or for the future, MIMICO is here to make the journey exciting and accessible for everyone. <br> Ready to wave hello in sign language? <br>Let’s get started — just click the button below and jump in!  
        <b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div>', unsafe_allow_html=True)
    col = st.columns(3)
    with col[1]:
        if st.button("Start Learning Now"):
            go_to("choose")
    st.markdown('</div>', unsafe_allow_html=True)


# Halaman 2: Pilih Menu
elif st.session_state.page == "choose":
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 120px;   /* atur lebar tombol */
            padding: 6px 12px; 
            font-size: 14px;
            border-radius : 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="back_from_choose"):
        go_back_menu()
        st.rerun()

    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='color:#AAB48B; text-shadow: 
                -1px -1px 0 #f6eedf,  
                1px -1px 0 #f6eedf,
                -1px 1px 0 #f6eedf,
                1px 1px 0 #ffffff;'>Pick Your Adventure!</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Pakai 3 kolom: kosong - isi - kosong, untuk sentralisasi
    spacer1, main_col, spacer2 = st.columns([1, 4, 1])
    with main_col:
        # Dalam kolom utama, buat dua kolom sejajar untuk tombol
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Detect", key="detect_button"):
                go_to("detect")  # ✅ ubah ini
                st.rerun()
        with col2:
            if st.button("Dictionary", key="dictionary_button"):
                go_to("dictionary")  # ✅ ubah ini
                st.rerun()
        with col3:
            if st.button("Quiz", key="quiz_button"):
                go_to("quiz")  # ✅ ubah ini
                st.rerun()
       
        st.markdown("""
            <style>
            div.stButton > button {
                font-size: 70px;
                padding: 90px 150px;
                border-radius: 90px;
                background-color: #8a8e75;
                color: #f6eedf;
                width: 100%;
                margin-top: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

# Halaman 3: Detect
elif st.session_state.page == "detect":
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 120px;   /* atur lebar tombol */
            padding: 6px 12px; 
            font-size: 14px;
            border-radius : 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="back_from_detect"):
        go_back_menu()
        st.rerun()
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style = 'color:#AAB48B; text-shadow: 
                -1px -1px 0 #f6eedf,  
                1px -1px 0 #f6eedf,
                -1px 1px 0 #f6eedf,'> Real-Time Hand Sign Detection</h1>
            </div>
        """,
        unsafe_allow_html=True
    )
    st.caption('Hey there! Turn on your webcam and show your hand gestures. MIMICO will recognize your sign language letters instantly.')
    run = st.checkbox('Start Webcam')

    if run:
        st.markdown(
        """
        <div style='text-align: center;'>
            <p style='font-size:18px;'>Point your hand at the camera and see what letters are read!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
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
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access camera. Make sure it is connected.")
        else:
            # Stream video 1 frame per loop
            while run:
                data_aux = []
                x_, y_ = [], []

                ret, frame = cap.read()
                if not ret:
                    st.warning("Can't read frame from camera.")
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

# Halaman 4: Dictionary
elif st.session_state.page == "dictionary":
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 120px;   /* atur lebar tombol */
            padding: 6px 12px; 
            font-size: 14px;
            border-radius : 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="back_from_dict"):
        go_back_menu()
        st.rerun()

    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='color: #AAB48B; text-shadow: 
                -1px -1px 0 #f6eedf,  
                1px -1px 0 #f6eedf,
                -1px 1px 0 #f6eedf,
                1px 1px 0 #ffffff;'>Sign-o-pedia</h1>
            <p>Wanna know how to sign a word? Browse our visual dictionary packed with gestures from A to Z! It’s your pocket-friendly sign language guide.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
        # Pakai 3 kolom: kosong - isi - kosong, untuk sentralisasi
    spacer1, main_col, spacer2 = st.columns([1, 3, 1])

    with main_col:
        # Dalam kolom utama, buat dua kolom sejajar untuk tombol
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Alphabet", key="alpha_button"):
                st.session_state.page = "alpha"

        with col2:
            if st.button("Number", key="num_button"):
                st.session_state.page = "num"
        
        st.markdown("""
            <style>
            div.stButton > button {
                font-size: 15px;
                padding: 10px 15px;
                border-radius: 5px;
                background-color: white;
                color: #52655A;
                width: 100%;
                margin-top: 5px;
                border: 2px solid #52655A;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s ease-in-out;
            }
            div.stButton > button:hover {
            background-color: #52655A;
            color: white;
            }
            </style>
        """, unsafe_allow_html=True)

# Halaman Alphabet
if st.session_state.page == "alpha":
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 120px;   /* atur lebar tombol */
            padding: 6px 12px; 
            font-size: 14px;
            border-radius : 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="back_from_alpha"):
        go_back_menu()
        st.rerun()

    st.markdown("""
    <h2 style=' 
        color:#ffffff;
        text-shadow:
            -1px -1px 0 #AAB48B,
            1px -1px 0 #AAB48B,
            -1px 1px 0 #AAB48B,
            1px 1px 0 #AAB48B;
    '>Alphabet Gestures</h2>
    """, unsafe_allow_html=True)

    # Daftar huruf contoh

    alphabet_gestures = [
        {"letter": "A", "image": "images/alphabet/A.jpg"},
        {"letter": "B", "image": "images/alphabet/B.jpg"},
        {"letter": "C", "image": "images/alphabet/C.jpg"},
        {"letter": "D", "image": "images/alphabet/D.jpg"},
        {"letter": "E", "image": "images/alphabet/E.jpg"},
        {"letter": "F", "image": "images/alphabet/F.jpg"},
        {"letter": "G", "image": "images/alphabet/G.jpg"},
        {"letter": "H", "image": "images/alphabet/H.jpg"},
        {"letter": "I", "image": "images/alphabet/I.jpg"},
        {"letter": "K", "image": "images/alphabet/K.jpg"},
        {"letter": "L", "image": "images/alphabet/L.jpg"},
    ]

    for item in alphabet_gestures:
        col1, col2 = st.columns([1, 2])  # kolom 1 untuk gambar, kolom 2 untuk huruf
        with col1:
            st.image(item["image"], width=250)
        with col2:
            st.markdown(f"""
                <div style='
                    padding: 8px 16px;
                    background-color: rgba(0, 0, 0, 0.1);
                    border-radius: 10px;
                    font-size: 24px;
                    font-weight: bold;
                    color: #ffffff;
                    display: flex;
                    align-items: center;
                    height: 100%;
                '>{item["letter"]}</div>
            """, unsafe_allow_html=True)

# Halaman Number
elif st.session_state.page == "num":
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 120px;   /* atur lebar tombol */
            padding: 6px 12px; 
            font-size: 14px;
            border-radius : 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="back_from_num"):
        go_back_menu()
        st.rerun()

    st.markdown("""
    <h2 style=' 
        color:#ffffff;
        text-shadow:
            -1px -1px 0 #AAB48B,
            1px -1px 0 #AAB48B,
            -1px 1px 0 #AAB48B,
            1px 1px 0 #AAB48B;
    '>Number Gestures</h2>
    """, unsafe_allow_html=True)

    # Daftar huruf contoh

    num_gestures = [
        {"letter": "0", "image": "images/number/0.jpg"},
        {"letter": "1", "image": "images/number/1.jpg"},
        {"letter": "2", "image": "images/number/2.jpg"},
        {"letter": "3", "image": "images/number/3.jpg"},
        {"letter": "4", "image": "images/number/4.jpg"},
        {"letter": "5", "image": "images/number/5.jpg"},
        {"letter": "6", "image": "images/number/6.jpg"},
    ]

    for item in num_gestures:
        col1, col2 = st.columns([1, 2])  # kolom 1 untuk gambar, kolom 2 untuk huruf
        with col1:
            st.image(item["image"], width=250)
        with col2:
            st.markdown(f"""
                <div style='
                    padding: 8px 16px;
                    background-color: rgba(0, 0, 0, 0.1);
                    border-radius: 10px;
                    font-size: 24px;
                    font-weight: bold;
                    color: #ffffff;
                    display: flex;
                    align-items: center;
                    height: 100%;
                '>{item["letter"]}</div>
            """, unsafe_allow_html=True)


# Halaman 5: Quiz
elif st.session_state.page == "quiz":
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 120px;   /* atur lebar tombol */
            padding: 6px 12px; 
            font-size: 14px;
            border-radius : 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="back_from_quiz"):
        go_back_menu()
        st.rerun()

    st.markdown("""
    <h1 style='
        color:#AAB48B; 
        text-align:center; 
        text-shadow: 
            -1px -1px 0 #f6eedf,  
            1px -1px 0 #f6eedf,
            -1px 1px 0 #f6eedf,
            1px 1px 0 #ffffff;
    '>
        Test Your Gesture Skills!
    </h1>
    """, unsafe_allow_html=True)

    # Inisialisasi variabel state
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_round" not in st.session_state:
        st.session_state.quiz_round = 0
    if "quiz_target" not in st.session_state:
        st.session_state.quiz_target = random.choice(list("ABCDEFGHIKLMNOPQRSTUVWXY"))

    TOTAL_ROUNDS = 5

    # Teks judul dengan shadow putih
    st.markdown(f"""
        <h3 style='
            color: #000000;
            text-shadow: 
                -1px -1px 0 ##312e23,
                1px -1px 0 ##312e23,
                -1px 1px 0 ##312e23,
                1px 1px 0 ##312e23;
        '>
            Round {st.session_state.quiz_round + 1} of {TOTAL_ROUNDS}
        </h3>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <p style='
            font-size: 20px;
            font-weight: bold;
            color: #000000;
            text-shadow: 
                -1px -1px 0 ##312e23,
                1px -1px 0 ##312e23,
                -1px 1px 0 ##312e23,
                1px 1px 0 ##312e23;
            background-color: rgba(234, 227, 211, 0.3);
        '>
            Try to imitate this letter with your hand:
        </p>
    """, unsafe_allow_html=True)

    # Tampilan huruf besar target gesture
    st.markdown(f"""
        <h2 style='
            text-align:center; 
            font-size:72px;
        '>{st.session_state.quiz_target}</h2>
    """, unsafe_allow_html=True)

    run_quiz = st.checkbox("Start Webcam for Quiz")

    if run_quiz:
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
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)

        timeout_start = time.time()
        timeout = 10  # deteksi dalam 10 detik

        detected_letter = None
        success = False

        while time.time() < timeout_start + timeout and not success:
            ret, frame = cap.read()
            if not ret:
                st.warning("Can't read frame from camera.")
                break

            data_aux = []
            x_, y_ = [], []

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
                    detected_letter = predicted_character

                    cv2.putText(frame, predicted_character, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    if predicted_character == st.session_state.quiz_target:
                        success = True
                        st.session_state.quiz_score += 1

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        st.success(f"Your Answer: {detected_letter or 'Not Detected'}")
        if detected_letter == st.session_state.quiz_target:
            st.balloons()
            st.success("🎉 Good Job! ")
        else:
            st.error("❌ Incorrect or illegible.")

        if st.button("Continue to the next question"):
            st.session_state.quiz_round += 1
            st.session_state.quiz_target = random.choice(list("ABCDEFGHIJKLMN"))
            if st.session_state.quiz_round >= TOTAL_ROUNDS:
                st.session_state.page = "quiz_result"

# Halaman hasil quiz
elif st.session_state.page == "quiz_result":
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 120px;   /* atur lebar tombol */
            padding: 6px 12px; 
            font-size: 14px;
            border-radius : 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="back_from_quizsc"):
        go_back_menu()
        st.rerun()
    score = st.session_state.quiz_score
    total = 5
    st.markdown("<h1 style='color:#E9F9D6; text-align:center;'>Quiz Finished!</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center;'>Your Score: {score} / {total}</h3>", unsafe_allow_html=True)

    if st.button("Try it Again"):
        st.session_state.quiz_score = 0
        st.session_state.quiz_round = 0
        st.session_state.page = "quiz"

    if st.button("Back to Menu"):
        st.session_state.quiz_score = 0
        st.session_state.quiz_round = 0
        go_to("choose")
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from gtts import gTTS
import threading
import os
from playsound import playsound

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WAVE_WINDOW = 6
WAVE_THRESHOLD = 0.12
ANNOUNCE_COOLDOWN = 1.5
TTS_FOLDER = 'tts_cache'
if not os.path.exists(TTS_FOLDER): os.makedirs(TTS_FOLDER)

GESTURE_TEXTS = {
    'wave': 'Halo! Senang bertemu denganmu.',
    'thumbs_up': 'Terima kasih!',
    'point_to_self': 'Nama saya Silo.',
    'no': 'Tidak.',
    'eat': 'Makan.',
    'run': 'Lari.',
    'drink': 'Minum.'
}

last_tts_time = 0
last_tts_text = ''

def speak(text):
    global last_tts_time, last_tts_text
    now = time.time()
    if text == last_tts_text and now - last_tts_time < ANNOUNCE_COOLDOWN:
        return  
    last_tts_text = text
    last_tts_time = now

    filename = os.path.join(TTS_FOLDER, f"{text}.mp3")
    if not os.path.exists(filename):
        tts = gTTS(text=text, lang='id', tld='com')
        tts.save(filename)
    threading.Thread(target=lambda: playsound(filename), daemon=True).start()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=2,
                                     min_detection_confidence=0.6,
                                     min_tracking_confidence=0.6)
        self.wrist_x_history = deque(maxlen=WAVE_WINDOW)

    def detect(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        gesture = None
        annotated = np.zeros_like(image)

        if results.multi_hand_landmarks:
            closest_hand = min(results.multi_hand_landmarks, key=lambda h: h.landmark[mp_hands.HandLandmark.WRIST].z)
            mp_drawing.draw_landmarks(annotated, closest_hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
            lm = closest_hand.landmark

            wrist = lm[mp_hands.HandLandmark.WRIST]
            self.wrist_x_history.append(wrist.x)

            if len(self.wrist_x_history) == self.wrist_x_history.maxlen:
                if max(self.wrist_x_history) - min(self.wrist_x_history) > WAVE_THRESHOLD:
                    gesture = 'wave'

            def finger_extended(tip, pip):
                return lm[tip].y < lm[pip].y

            if (lm[mp_hands.HandLandmark.THUMB_TIP].y < lm[mp_hands.HandLandmark.THUMB_IP].y and
                not any(finger_extended(f, p) for f, p in [
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
                ])):
                gesture = 'thumbs_up'

            if (finger_extended(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                not any(finger_extended(f, p) for f, p in [
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
                ])):
                gesture = 'point_to_self'

            if all(not finger_extended(f, p) for f, p in [
                (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
            ]):
                gesture = 'no'

        else:
            self.wrist_x_history.append(0.5)

        return annotated, gesture

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Aplikasi By Silo Kusuma')

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.text_var = tk.StringVar(value='(Tunjukkan tanganmu)')
        self.text_label = tk.Label(root, textvariable=self.text_var, font=('Helvetica', 18), bg='black', fg='white')
        self.text_label.place(x=10, y=10)

        self.btn_quit = tk.Button(root, text='Keluar', command=self.on_close)
        self.btn_quit.pack(padx=8, pady=6)

        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.detector = GestureDetector()
        self.running = True
        self.update_video()

        root.protocol('WM_DELETE_WINDOW', self.on_close)

    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.text_var.set('Tidak dapat mengakses kamera')
            return

        frame = cv2.flip(frame, 1)
        annotated, gesture = self.detector.detect(frame)

        if gesture:
            text = GESTURE_TEXTS.get(gesture, '')
            if text:
                self.text_var.set(text)
                speak(text)

        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(5, self.update_video) 

    def on_close(self):
        self.running = False
        time.sleep(0.05)
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.quit()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry(f'{FRAME_WIDTH}x{FRAME_HEIGHT+80}')
    app = App(root)
    root.mainloop()
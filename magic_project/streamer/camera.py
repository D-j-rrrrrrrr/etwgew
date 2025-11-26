import cv2
import threading
import time
import mediapipe as mp
import numpy as np

# --- Gesture Constants ---
GESTURE_SWIPE_THRESHOLD_X = 0.15
GESTURE_SWIPE_MAX_TIME = 0.5
GESTURE_COOLDOWN = 1.0

class VideoCamera(object):
    def __init__(self):
        # Open the camera (Using index 1 as per your setup)
        self.video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        # --- MediaPipe Setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # --- Filter State ---
        self.filter_mode = 0 # 0: Normal, 1: Cartoon, 2: Grayscale
        
        # --- Gesture Variables ---
        self.gesture_start_x = None
        self.gesture_start_time = None
        self.gesture_cooldown_end = 0.0

    def __del__(self):
        self.video.release()

    # --- Filter Logic ---
    def cartoonize_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.medianBlur(gray, 5) 
        edges = cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=5)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        color_filtered = cv2.bilateralFilter(frame, d=15, sigmaColor=250, sigmaSpace=250)
        return cv2.bitwise_and(color_filtered, edges_bgr)

    def grayscale_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def is_fist(self, hand_landmarks):
        try:
            fingers_curled = [
                (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
                (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
                (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
            ]
            for tip_idx, pip_idx in fingers_curled:
                if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[pip_idx].y:
                    return False
            # Thumb check
            if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y:
                return False
            return True
        except: return False

    def process_gestures(self, results):
        if time.time() < self.gesture_cooldown_end: return
        if not results.multi_hand_landmarks:
            self.gesture_start_x = None
            return

        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Check Fist
        if self.is_fist(hand_landmarks):
            if self.filter_mode != 0:
                self.filter_mode = 0
                self.gesture_cooldown_end = time.time() + GESTURE_COOLDOWN
            self.gesture_start_x = None
            return

        # Check Swipe
        wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
        
        if self.gesture_start_x is None:
            self.gesture_start_x = wrist_x
            self.gesture_start_time = time.time()
            return

        elapsed = time.time() - self.gesture_start_time
        delta_x = wrist_x - self.gesture_start_x
        
        if elapsed > GESTURE_SWIPE_MAX_TIME:
            self.gesture_start_x = None
            return

        # Swipe Logic (Non-Mirrored)
        if delta_x < -GESTURE_SWIPE_THRESHOLD_X:
            self.filter_mode = 1 # Cartoon
            self.gesture_cooldown_end = time.time() + GESTURE_COOLDOWN
            self.gesture_start_x = None
        elif delta_x > GESTURE_SWIPE_THRESHOLD_X:
            self.filter_mode = 2 # Grayscale
            self.gesture_cooldown_end = time.time() + GESTURE_COOLDOWN
            self.gesture_start_x = None

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None

        # --- 1. Hand Tracking ---
        # Note: We commented out flip in your Tkinter app, so we keep it raw here too
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        
        # --- 2. Gesture Logic ---
        self.process_gestures(results)

        # --- 3. Filter Application ---
        if self.filter_mode == 1:
            image = self.cartoonize_frame(image)
        elif self.filter_mode == 2:
            image = self.grayscale_frame(image)

        # --- 4. Draw Landmarks ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                )

        # --- 5. Encode for Web ---
        # We need to turn the image into bytes to send over the internet
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
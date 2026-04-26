import os
import cv2
import json
import csv
import numpy as np
from datetime import datetime, timedelta

FACES_DIR = "faces"
USERS_FILE = "users.json"
ATTENDANCE_FILE = "attendance.csv"
TRAINER_FILE = "trainer.yml"

class AttendanceSystem:
    def __init__(self):
        self.users = {}
        self.label_to_name = {}
        
        # Ensure directories and files exist
        if not os.path.exists(FACES_DIR):
            os.makedirs(FACES_DIR)
        
        if not os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'w') as f:
                json.dump({}, f)
                
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Time", "Date", "Period", "Status"])
                
        # Initialize Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize LBPH Recognizer
        try:
            # radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            print("Error: opencv-contrib-python not found. Falling back to mock.")
            self.recognizer = None

        self.load_users()
        self.load_faces()

    def load_users(self):
        with open(USERS_FILE, 'r') as f:
            try:
                self.users = json.load(f)
            except json.JSONDecodeError:
                self.users = {}

    def save_users(self):
        with open(USERS_FILE, 'w') as f:
            json.dump(self.users, f, indent=4)

    def prepare_face_roi(self, gray_roi):
        """Standardize face ROI for training and prediction."""
        # 1. Resize to a consistent resolution (e.g., 200x200)
        resized = cv2.resize(gray_roi, (200, 200), interpolation=cv2.INTER_LANCZOS4)
        # 2. Histogram Equalization to normalize lighting
        equalized = cv2.equalizeHist(resized)
        return equalized

    def train_model(self):
        if not self.recognizer:
            return

        faces = []
        labels = []
        current_id = 0
        self.label_to_name = {}

        # Iterate through user folders
        for username in sorted(os.listdir(FACES_DIR)):
            user_path = os.path.join(FACES_DIR, username)
            if not os.path.isdir(user_path):
                continue
            
            current_id += 1
            self.label_to_name[current_id] = username
            
            samples_count = 0
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in detected_faces:
                    face_roi = gray[y:y+h, x:x+w]
                    # Quality Control: Apply Standardization
                    processed_face = self.prepare_face_roi(face_roi)
                    faces.append(processed_face)
                    labels.append(current_id)
                    samples_count += 1

            if samples_count > 0:
                print(f"[LBPH] Prepared {samples_count} samples for user: {username}")

        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(TRAINER_FILE)
            print(f"[LBPH] SUCCESS: Trained on {len(self.label_to_name)} users with {len(faces)} total samples.")
        else:
            print("[LBPH] WARNING: No face samples detected in training set.")

    def load_faces(self):
        if not self.recognizer:
            return

        if os.path.exists(TRAINER_FILE) and os.path.getsize(TRAINER_FILE) > 0:
            try:
                self.recognizer.read(TRAINER_FILE)
                # Re-build label mapping from directory structure
                current_id = 0
                for username in sorted(os.listdir(FACES_DIR)):
                    user_path = os.path.join(FACES_DIR, username)
                    if os.path.isdir(user_path):
                        current_id += 1
                        self.label_to_name[current_id] = username
                print(f"[LBPH] Model loaded with {len(self.label_to_name)} identities.")
            except Exception as e:
                print(f"[LBPH] Model load failed: {e}. Attempting recovery...")
                self.train_model()
        else:
            self.train_model()

    def get_period(self, username, current_datetime=None):
        if current_datetime is None:
            current_datetime = datetime.now()
            
        start_time_str = self.users.get(username, "09:00")
        try:
            stObj = datetime.strptime(start_time_str, "%H:%M").time()
        except ValueError:
            stObj = datetime.strptime("09:00", "%H:%M").time()
            
        dt_today = datetime.combine(current_datetime.date(), stObj)
        
        shifts = [dt_today - timedelta(days=1), dt_today, dt_today + timedelta(days=1)]
        
        for s_start in shifts:
            s_end = s_start + timedelta(hours=8)
            if s_start <= current_datetime < s_end:
                time_diff = current_datetime - s_start
                hours_in = time_diff.total_seconds() / 3600.0
                period = int(hours_in) + 1
                if period > 8: period = 8
                return period
        
        return "OUT_OF_HOURS"

    def log_attendance(self, username, period):
        now = datetime.now()
        dt_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        already_logged = False
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["Name"] == username and row["Date"] == dt_str and str(row["Period"]) == str(period):
                        already_logged = True
                        break
        
        if not already_logged:
            status = "SUCCESS" if str(period) != "OUT_OF_HOURS" else "OUT_OF_HOURS"
            with open(ATTENDANCE_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([username, time_str, dt_str, period, status])
            return True, status
        return False, "ALREADY_LOGGED"

    def process_frame(self, frame):
        # Resize frame for detection performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_locations = []
        recognized_names = []
        
        for (x, y, w, h) in faces:
            # Scale back to original (not really needed for ROI, but for box coordinates)
            top, right, bottom, left = y, x + w, y + h, x
            face_locations.append((top, right, bottom, left))
            
            name = "Unknown"
            if self.recognizer and len(self.label_to_name) > 0:
                roi_gray = gray[y:y+h, x:x+w]
                # Apply same optimization (Resize + Histogram Equalize)
                processed_roi = self.prepare_face_roi(roi_gray)
                
                label_id, confidence = self.recognizer.predict(processed_roi)
                
                # LBPH: Lower distance (confidence) means better match.
                # Threshold of 100.0 is usually safe; higher = more permissive.
                if confidence < 100:
                    name = self.label_to_name.get(label_id, "Unknown")
                    print(f"[LBPH] MATCH: {name} (Distance: {confidence:.2f})")
                else:
                    print(f"[LBPH] UNKNOWN (Closest: {self.label_to_name.get(label_id, 'None')} @ Distance: {confidence:.2f})")
                
            recognized_names.append(name)
            
        return face_locations, recognized_names

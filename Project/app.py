from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
from functools import wraps
import cv2
import os
import csv
import numpy as np
import base64
from recognition_logic import AttendanceSystem

app = Flask(__name__)
app.secret_key = 'super_secret_key_attendance_v3'

attendance_system = AttendanceSystem()

# Global events for UI feedback
last_event = {
    "name": None,
    "period": None,
    "status": None
}

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['password'] == 'admin':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid Password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@admin_required
def index():
    # Read attendance data
    attendance_data = []
    today_count = 0
    from datetime import datetime
    now_date = datetime.now().strftime("%Y-%m-%d")
    
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            attendance_data = list(reader)
            for row in attendance_data:
                # Assuming Date is physically at index 2 (Name, Time, Date, Period, Status)
                if len(row) > 2 and row[2] == now_date:
                    today_count += 1
                    
            # Reverse to show newest on top
            attendance_data.reverse()
            
    return render_template('index.html', data=attendance_data, todays_checkins=today_count)

@app.route('/manage')
@admin_required
def manage():
    faces_data = []
    faces_dir = "faces"
    if os.path.exists(faces_dir):
        for username in os.listdir(faces_dir):
            user_path = os.path.join(faces_dir, username)
            if os.path.isdir(user_path):
                # Just show the first image as a preview
                user_images = [f for f in os.listdir(user_path) if f.endswith('.jpg') or f.endswith('.png')]
                if user_images:
                    preview_file = os.path.join(username, user_images[0])
                    period_start = attendance_system.users.get(username, "09:00")
                    faces_data.append({"name": username, "file": preview_file, "start": period_start})
    return render_template('manage.html', faces=faces_data)

@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    name = data.get('name')
    start_time = data.get('start_time', '09:00')
    images = data.get('images', []) # Expecting a list of 10 base64 images
    
    if not name or not images:
        return jsonify({"error": "Missing name or images"}), 400
        
    try:
        user_dir = os.path.join("faces", name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        for i, image_data in enumerate(images):
            # Decode base64 image
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Save image
            img_path = os.path.join(user_dir, f"{i+1}.jpg")
            cv2.imwrite(img_path, img)
        
        # Update users config
        attendance_system.users[name] = start_time
        attendance_system.save_users()
        
        # Trigger LBPH Training
        attendance_system.train_model()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

def gen_frames():
    global last_event
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face_locations, recognized_names = attendance_system.process_frame(frame)
            
            for (top, right, bottom, left), name in zip(face_locations, recognized_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                if name != "Unknown":
                    # Determine period
                    period = attendance_system.get_period(name)
                    
                    if period == "OUT_OF_HOURS":
                        attendance_system.log_attendance(name, "OUT_OF_HOURS")
                        last_event = {"name": name, "period": "OUT_OF_HOURS", "status": "OUT_OF_HOURS"}
                        label = f"{name} (OUT OF HOURS)"
                        color = (0, 0, 255)
                    else:
                        logged, status = attendance_system.log_attendance(name, period)
                        if status == "SUCCESS":
                            last_event = {"name": name, "period": period, "status": "NEW"}
                        elif status == "ALREADY_LOGGED":
                            # Don't trigger sound again, but update UI visually if desired
                            pass
                        
                        label = f"{name} (P{period})"
                        color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/poll')
def api_poll():
    global last_event
    resp = jsonify(last_event)
    # Clear after returning so it doesn't repeatedly trigger
    last_event = {"name": None, "period": None, "status": None}
    return resp

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

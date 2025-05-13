from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from flask_socketio import SocketIO
import cv2
import numpy as np
import mediapipe as mp
import base64
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
socketio = SocketIO(app)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to extract motion-based features
def extract_motion_features(frame, prev_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    motion_intensity = np.count_nonzero(motion_mask)
    return motion_intensity

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.prev_frame = None
        self.activities = {
            1: 'Walking', 2: 'Running', 3: 'Crawling', 
            4: 'Sitting', 5: 'Standing', 6: 'Eating', 
            7: 'Dancing', 8: 'Jumping', 9: 'Stretching'
        }

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None, None, "No frame"

        if self.prev_frame is None:
            self.prev_frame = frame
            return frame, None, "Initializing"

        motion_intensity = extract_motion_features(frame, self.prev_frame)

        # Determine activity based on motion intensity
        if motion_intensity < 1500:
            activity_label = 5  # Standing
        elif motion_intensity < 5000:
            activity_label = 3  # Crawling
        elif motion_intensity < 10000:
            activity_label = 1  # Walking
        elif motion_intensity < 20000:
            activity_label = 5  # Standing
        elif motion_intensity < 30000:
            activity_label = 6  # Eating
        elif motion_intensity < 50000:
            activity_label = 8  # Jumping
        else:
            activity_label = 7  # Dancing

        # Pose estimation
        img = cv2.resize(frame, (700, 600))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        # Create pose estimation visualization
        pose_img = np.zeros_like(img)
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                pose_img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        self.prev_frame = frame
        return img, pose_img, self.activities[activity_label]

camera = None

# Route for landing page
@app.route('/')
def landing():
    return render_template('landing.html')
    \

# Route for home page
@app.route('/home')
def home():
    return render_template('home.html')

# Route for services page
@app.route('/services')
def services():
    return render_template('services.html')

# Route for contact page
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        return redirect(url_for('contact'))
    return render_template('contact.html')

# Route for input page
@app.route('/input')
def input():
    return render_template('input.html')

# Route for index page (camera feed)
@app.route('/index')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video and make prediction
        prediction = process_video(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({'prediction': prediction})
    
    return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Initialize variables
    prev_frame = None
    activities_count = {}
    frame_count = 0
    
    # Dictionary for activities (same as in VideoCamera class)
    activities = {
        1: 'Walking', 2: 'Running', 3: 'Crawling', 
        4: 'Sitting', 5: 'Standing', 6: 'Eating', 
        7: 'Dancing', 8: 'Jumping', 9: 'Stretching'
    }
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            frame_count += 1
            
            if prev_frame is None:
                prev_frame = frame
                continue
            
            # Get motion intensity
            motion_intensity = extract_motion_features(frame, prev_frame)
            
            # Determine activity based on motion intensity (same logic as in VideoCamera)
            if motion_intensity < 1500:
                activity_label = 5  # Standing
            elif motion_intensity < 5000:
                activity_label = 3  # Crawling
            elif motion_intensity < 10000:
                activity_label = 1  # Walking
            elif motion_intensity < 20000:
                activity_label = 5  # Standing
            elif motion_intensity < 30000:
                activity_label = 6  # Eating
            elif motion_intensity < 50000:
                activity_label = 8  # Jumping
            else:
                activity_label = 7  # Dancing
            
            # Count occurrences of each activity
            activity_name = activities[activity_label]
            activities_count[activity_name] = activities_count.get(activity_name, 0) + 1
            
            prev_frame = frame
            
        # Find the most common activity
        if activities_count:
            most_common_activity = max(activities_count.items(), key=lambda x: x[1])[0]
            confidence = (activities_count[most_common_activity] / frame_count) * 100
            
            return f"Detected Activity: {most_common_activity} (Confidence: {confidence:.1f}%)"
        else:
            return "No activity detected"
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return "Error processing video"
        
    finally:
        video.release()
        pose.close()

def gen_frames():
    global camera
    while True:
        frame, pose_frame, activity = camera.get_frame()
        if frame is None:
            break
            
        # Convert both frames to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        if pose_frame is not None:
            ret, pose_buffer = cv2.imencode('.jpg', pose_frame)
            pose_bytes = pose_buffer.tobytes()
        else:
            pose_bytes = None

        # Emit the activity through Socket.IO
        socketio.emit('activity_update', {'activity': activity})
        
        # Yield the frame bytes
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return "Camera started"

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.__del__()
        camera = None
    return "Camera stopped"

if __name__ == '__main__':
    socketio.run(app, debug=True) 

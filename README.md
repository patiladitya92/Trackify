# Real-time Activity Recognition Web Application

This web application performs real-time activity recognition using computer vision and pose estimation. It can detect various activities such as walking, running, crawling, sitting, standing, eating, dancing, and jumping.

## Features

- Real-time activity detection using motion analysis
- Pose estimation visualization
- Modern web interface with live camera feed
- Start/Stop functionality
- Real-time activity display

## Prerequisites

- Python 3.7 or higher
- Webcam
- Modern web browser

## Installation

1. Navigate to the application directory:
```bash
cd D:\activity_recognition_app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Click the "Start Prediction" button to begin activity recognition
4. The detected activity will be displayed in real-time
5. Click "Stop" to end the session

## Technical Details

The application uses:
- Flask for the web server
- OpenCV for video processing
- MediaPipe for pose estimation
- Socket.IO for real-time updates
- TailwindCSS for styling

## Notes

- Make sure your webcam is properly connected and accessible
- Allow camera access when prompted by your browser
- For best results, ensure good lighting and clear view of your movements 
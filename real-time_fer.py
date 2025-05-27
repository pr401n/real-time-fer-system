import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Real-time Emotion Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the video feed smaller and scrollable
st.markdown("""
    <style>
    .video-container {
        max-height: 400px;
        overflow-y: auto;
        border: 2px solid #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .st-emotion-caption {
        font-size: 0.8rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🎭 Real-time Emotion Detection")
st.markdown("""
    <div style='margin-bottom: 20px;'>
        This application detects emotions in real-time using your webcam.
        The model can detect 7 emotional states: Angry, Disgusted, Fearful, 
        Happy, Sad, Surprised, and Neutral.
    </div>
    """, unsafe_allow_html=True)

# Load the model and resources
@st.cache_resource
def load_emotion_model():
    return load_model('face_model.h5')

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    model_best = load_emotion_model()
    face_cascade = load_face_cascade()
except Exception as e:
    st.error(f"Error loading model or cascade: {str(e)}")
    st.stop()

class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Control panel in sidebar
with st.sidebar:
    st.header("Controls")
    run_detection = st.checkbox("Start Detection", value=False)
    detection_confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.7, 0.05)
    frame_width = st.slider("Frame Width", 300, 800, 500, 50)
    show_fps = st.checkbox("Show FPS", value=True)
    
    st.markdown("---")
    st.header("Detection Settings")
    scale_factor = st.slider("Scale Factor", 1.01, 1.5, 1.3, 0.05)
    min_neighbors = st.slider("Min Neighbors", 1, 10, 5)
    min_size = st.slider("Min Face Size", 20, 100, 30)

# Create a container for the video feed
video_container = st.container()
frame_placeholder = video_container.empty()
fps_placeholder = video_container.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open webcam. Please check your camera permissions.")
    st.stop()

# FPS calculation
prev_time = 0
fps = 0

while run_detection:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from webcam")
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size)
    )

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face image to the required input size for the model
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        # Predict emotion using the loaded model
        try:
            predictions = model_best.predict(face_image, verbose=0)
            max_prob = np.max(predictions)
            if max_prob > detection_confidence:
                emotion_label = class_names[np.argmax(predictions)]
                # Display the emotion label on the frame
                cv2.putText(frame, f'{emotion_label} ({max_prob:.2f})', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            continue

    # Resize frame to make it smaller
    height, width = frame.shape[:2]
    new_height = int(height * frame_width / width)
    resized_frame = cv2.resize(frame, (frame_width, new_height))

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame in a scrollable container
    with video_container:
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=False)
        if show_fps:
            fps_placeholder.caption(f"FPS: {fps:.1f} | Detected faces: {len(faces)}")

# Release resources when not running
if not run_detection:
    cap.release()
    cv2.destroyAllWindows()
    with video_container:
        frame_placeholder.empty()
        fps_placeholder.empty()
    st.info("Emotion detection is stopped. Enable the checkbox to start.")

# Add documentation in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Click 'Start Detection' to begin
    2. Adjust settings as needed:
       - Increase 'Detection Confidence' for more accurate results
       - Adjust 'Frame Width' to change display size
       - Tune detection parameters if needed
    3. View your detected emotions in real-time
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses:
    - OpenCV for face detection
    - A pre-trained CNN for emotion classification
    - Streamlit for the web interface
    """)

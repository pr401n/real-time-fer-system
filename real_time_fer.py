###
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set page config
st.set_page_config(
    page_title="Real-time FER",
)

# Set page title
st.title("🎭 Real-time FER")
st.markdown("""
    <div style='margin-bottom: 20px;'>
        This application detects emotions in real-time using your webcam.
        The model can detect 7 emotional states: Angry, Disgusted, Fearful, 
        Happy, Sad, Surprised, and Neutral.
    </div>
    """, unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_emotion_model():
    return load_model('face_model.h5')

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_best = load_emotion_model()
face_cascade = load_face_cascade()
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start/stop button
run_detection = st.checkbox("Start Video Feed")

# Placeholder for video frame
frame_placeholder = st.empty()

# Initialize webcam
cap = None

if run_detection:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        run_detection = False

while run_detection:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each face
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        predictions = model_best.predict(face_image)
        emotion_label = class_names[np.argmax(predictions)]

        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    # Display frame in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

    # Add small delay and rerun to maintain video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources when stopped
if not run_detection and cap is not None:
    cap.release()
    cv2.destroyAllWindows()
    frame_placeholder.empty()


# Add documentation in sidebar
with st.sidebar:
    st.markdown("### How to Use")
    st.markdown("""
    1. Click 'Start Video Feed' to begin
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
    - A Custom-trained CNN for emotion classification
    - Streamlit for the web interface
    """)

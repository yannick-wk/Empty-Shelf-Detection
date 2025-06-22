#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Empty Shelf Detector - Streamlit App

This application allows using the improved empty shelf detection model in real-time
with camera input or custom image uploads.
"""

import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import pandas as pd
from datetime import datetime, timedelta

# Import utility modules
from model_utils import load_model
from evaluation_utils import visualize_predictions

# Set page configuration
st.set_page_config(
    page_title="Empty Shelf Detector",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define model paths
BASE_DIR = "/home/yannick/tf-nn/PERSONAL"
IMPROVED_MODEL_PATH = f"{BASE_DIR}/runs/detect/yolov8m_improved_empty_shelf/weights/best.pt"
IMPROVED_MODEL2_PATH = f"{BASE_DIR}/runs/detect/yolov8m_improved_empty_shelf_dataset2/weights/best.pt"
ORIGINAL_MODEL_PATH = f"{BASE_DIR}/runs/detect/yolov8n_empty_shelf_detector_v1/weights/best.pt"
DEFAULT_MODEL_PATH = f"{BASE_DIR}/yolov8m.pt"  # Fallback to base model if trained models not found

# Confidence threshold slider in sidebar
def sidebar_settings():
    st.sidebar.title("Model Settings")
    confidence = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detection boxes"
    )
    
    # Model selection
    available_models = []
    model_paths = {}
    
    # Check which models are available
    if os.path.exists(IMPROVED_MODEL_PATH):
        available_models.append("Improved Model (YOLOv8m)")
        model_paths["Improved Model (YOLOv8m)"] = IMPROVED_MODEL_PATH
    
    if os.path.exists(IMPROVED_MODEL2_PATH):
        available_models.append("Improved Model 2 (YOLOv8m Dataset2)")
        model_paths["Improved Model 2 (YOLOv8m Dataset2)"] = IMPROVED_MODEL2_PATH
    
    if os.path.exists(ORIGINAL_MODEL_PATH):
        available_models.append("Original Model (YOLOv8n)")
        model_paths["Original Model (YOLOv8n)"] = ORIGINAL_MODEL_PATH
    
    # Always add base model as fallback
    available_models.append("Base Model (YOLOv8m)")
    model_paths["Base Model (YOLOv8m)"] = DEFAULT_MODEL_PATH
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=0,
        help="Choose which model to use for detection"
    )
    
    use_tta = st.sidebar.checkbox(
        "Use Test-Time Augmentation",
        value=False,
        help="Enable test-time augmentation for better accuracy (slower)"
    )
    
    return confidence, model_paths[selected_model], use_tta

# Load the model
@st.cache_resource
def get_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to perform detection on an image
def detect_empty_shelves(model, image, conf_threshold=0.25, use_tta=False):
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert BGR to RGB if needed
    if image.shape[-1] == 3 and not isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform prediction
    start_time = time.time()
    results = model.predict(image, conf=conf_threshold, augment=use_tta, verbose=False)[0]
    inference_time = time.time() - start_time
    
    # Get bounding boxes, confidences, and class IDs
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    
    # Create a copy of the image for drawing
    img_with_boxes = image.copy() if isinstance(image, np.ndarray) else np.array(image).copy()
    
    # Draw bounding boxes
    locations = []
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        
        # Store location information
        locations.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width': x2 - x1, 'height': y2 - y1,
            'center_x': (x1 + x2) // 2, 'center_y': (y1 + y2) // 2,
            'confidence': float(conf)
        })
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add confidence text
        text = f"Empty Shelf: {conf:.2f}"
        cv2.putText(img_with_boxes, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Create detection record for analytics
    detection_record = {
        'timestamp': datetime.now(),
        'empty_count': len(boxes),
        'inference_time': inference_time,
        'confidence_avg': float(np.mean(confidences)) if len(confidences) > 0 else 0,
        'confidence_min': float(np.min(confidences)) if len(confidences) > 0 else 0,
        'confidence_max': float(np.max(confidences)) if len(confidences) > 0 else 0,
        'locations': locations,
        'image_shape': image.shape[:2]  # Store height, width
    }
    
    return img_with_boxes, boxes, confidences, inference_time, detection_record

# Function for camera capture
def camera_capture(model, conf_threshold, use_tta):
    st.title("Real-time Empty Shelf Detection")
    
    # Camera input
    camera_placeholder = st.empty()
    camera_stopped = st.button("Stop Camera")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please make sure your camera is connected and not in use by another application.")
        return
    
    # Frame rate calculation
    frame_times = []
    
    # Capture interval control
    save_to_analytics = st.checkbox("Save detection data to analytics", value=True)
    capture_interval = st.slider("Analytics capture interval (seconds)", 1, 30, 5)
    last_capture_time = time.time() - capture_interval  # Initialize to capture first frame
    
    while not camera_stopped:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        result_img, boxes, confidences, inference_time, detection_record = detect_empty_shelves(
            model, frame_rgb, conf_threshold, use_tta
        )
        
        # Calculate FPS
        frame_times.append(inference_time)
        if len(frame_times) > 30:  # Keep only last 30 frames for FPS calculation
            frame_times.pop(0)
        
        avg_inference_time = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Add FPS counter to image
        cv2.putText(result_img, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection count
        cv2.putText(result_img, f"Detections: {len(boxes)}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        camera_placeholder.image(result_img, channels="RGB", use_container_width=True)
        
        # Save detection data to analytics at specified interval
        current_time = time.time()
        if save_to_analytics and (current_time - last_capture_time) >= capture_interval:
            st.session_state.detection_history.append(detection_record)
            last_capture_time = current_time
    
    # Release webcam
    cap.release()
    st.info("Camera stopped")

# Function for image upload
def image_upload(model, conf_threshold, use_tta):
    st.title("Empty Shelf Detection - Image Upload")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
        # Perform detection
        result_img, boxes, confidences, inference_time, detection_record = detect_empty_shelves(
            model, image, conf_threshold, use_tta
        )
        
        # Display results
        st.subheader("Detection Results")
        st.image(result_img, use_container_width=True)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Inference Time", f"{inference_time:.3f} seconds")
        with col2:
            st.metric("Empty Shelves Detected", len(boxes))
        
        # Display detection details
        if len(boxes) > 0:
            st.subheader("Detection Details")
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                st.write(f"Detection {i+1}: Confidence = {conf:.4f}")
        
        # Save detection to history
        save_to_analytics = st.checkbox("Save to analytics dashboard", value=True)
        if save_to_analytics:
            st.session_state.detection_history.append(detection_record)
            st.success(f"Detection saved to analytics dashboard. Total records: {len(st.session_state.detection_history)}")

# Function for analytics dashboard
def analytics_dashboard(detection_history):
    st.title("Empty Shelf Analytics Dashboard")
    
    if not detection_history:
        st.info("No detection data available yet. Run some detections first.")
        return
    
    # Convert detection history to DataFrame for easier analysis
    df = pd.DataFrame(detection_history)
    
    # Summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Scans", 
            len(detection_history),
            help="Total number of detection sessions"
        )
    
    with col2:
        avg_empty = df['empty_count'].mean()
        st.metric(
            "Avg. Empty Shelves", 
            f"{avg_empty:.1f}",
            help="Average number of empty shelves detected per scan"
        )
    
    with col3:
        detection_rate = (df['empty_count'] > 0).mean() * 100
        st.metric(
            "Detection Rate", 
            f"{detection_rate:.1f}%",
            help="Percentage of scans with at least one empty shelf detected"
        )
    
    with col4:
        avg_time = df['inference_time'].mean() * 1000  # Convert to ms
        st.metric(
            "Avg. Processing Time", 
            f"{avg_time:.1f} ms",
            help="Average time to process one image"
        )
    
    # Time series chart
    st.subheader("Empty Shelf Detections Over Time")
    
    # Filter options
    time_periods = ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
    selected_period = st.selectbox("Time Period", time_periods)
    
    # Filter data based on selected period
    filtered_df = df.copy()
    now = datetime.now()
    
    if selected_period == "Last 24 Hours":
        filtered_df = df[df['timestamp'] > (now - timedelta(days=1))]
    elif selected_period == "Last 7 Days":
        filtered_df = df[df['timestamp'] > (now - timedelta(days=7))]
    elif selected_period == "Last 30 Days":
        filtered_df = df[df['timestamp'] > (now - timedelta(days=30))]
    
    if len(filtered_df) > 0:
        # Time series chart
        chart_df = filtered_df.set_index('timestamp')
        st.line_chart(chart_df['empty_count'])
        
        # Distribution of detections
        st.subheader("Distribution of Empty Shelf Detections")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create histogram of empty shelf counts
        ax.hist(filtered_df['empty_count'], bins=range(0, max(filtered_df['empty_count']) + 2), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Number of Empty Shelves')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        # Detection locations (if available)
        if 'locations' in filtered_df.columns and any(filtered_df['locations'].apply(len) > 0):
            st.subheader("Common Empty Shelf Locations")
            # Process location data (implementation would depend on how locations are stored)
    else:
        st.info(f"No data available for the selected period: {selected_period}")
    
    # Raw data table
    st.subheader("Detection History")
    display_df = filtered_df[['timestamp', 'empty_count', 'inference_time']].copy()
    display_df['inference_time'] = display_df['inference_time'].apply(lambda x: f"{x*1000:.1f} ms")
    display_df.columns = ['Timestamp', 'Empty Shelves', 'Processing Time']
    st.dataframe(display_df)

def main():
    # Initialize session state for detection history
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subheader {
            font-size: 1.5rem;
            color: #34495e;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<h1 class="main-header">Empty Shelf Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center'>
        Detect empty shelves in retail environments using advanced computer vision
        </p>
    """, unsafe_allow_html=True)
    
    # Get sidebar settings
    conf_threshold, model_path, use_tta = sidebar_settings()
    
    # Load model
    model = get_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Mode selection
    st.sidebar.title("Detection Mode")
    mode = st.sidebar.radio(
        "Select Input Mode",
        options=["Camera", "Upload Image", "Analytics Dashboard"]
    )
    
    # Run the selected mode
    if mode == "Camera":
        camera_capture(model, conf_threshold, use_tta)
    elif mode == "Upload Image":
        image_upload(model, conf_threshold, use_tta)
    else:
        analytics_dashboard(st.session_state.detection_history)
    
    # About section
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a YOLOv8-based model to detect empty shelves in retail environments. "
        "The model was trained on a custom dataset and optimized for accuracy and speed."
    )

if __name__ == "__main__":
    main()

# Empty Shelf Detection

A computer vision project for detecting empty shelves in retail environments using YOLOv8 models.

## Overview

This project provides a solution for detecting empty shelves in retail stores using deep learning. It includes a Streamlit web application for real-time detection using camera input or uploaded images.

## Features

- Real-time empty shelf detection using webcam
- Image upload and analysis
- Adjustable confidence threshold for detections
- Analytics dashboard for detection history
- Support for multiple YOLOv8 models
- TTA (Test Time Augmentation) option for improved accuracy

## Requirements

- Python 3.8+
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- Other dependencies listed in `requirements_app.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yannick-wk/Empty-Shelf-Detection.git
   cd Empty-Shelf-Detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements_app.txt
   ```

## Usage

Run the Streamlit application:

```
streamlit run empty_shelf_detector_app.py
```

### Model Settings

- Adjust detection confidence threshold in the sidebar
- Choose between different trained models
- Enable/disable Test Time Augmentation for improved accuracy

## Project Structure

- `empty_shelf_detector_app.py`: Main Streamlit application
- `model_utils.py`: Utilities for loading and managing models
- `evaluation_utils.py`: Functions for evaluating and visualizing predictions
- `ensemble_utils.py`: Ensemble model implementation
- `augmentation_utils.py`: Image augmentation utilities
- `data/`: Dataset directory
- `runs/`: Directory containing trained model weights and results

## Acknowledgements

- YOLOv8 by Ultralytics
- Streamlit for the web application framework

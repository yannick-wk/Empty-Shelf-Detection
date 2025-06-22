#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Augmentation Utilities for Improved Empty Shelf Detection

This module provides functions for advanced data augmentation techniques
to improve model robustness and performance.
"""

import os
import yaml
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    """
    Adjust brightness and contrast of an image
    
    Args:
        image (numpy.ndarray): Input image (RGB)
        brightness (float): Brightness factor (>1 = brighter, <1 = darker)
        contrast (float): Contrast factor (>1 = more contrast, <1 = less contrast)
        
    Returns:
        numpy.ndarray: Adjusted image
    """
    # Convert to PIL Image
    pil_img = Image.fromarray(image.astype('uint8'))
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)
    
    # Convert back to numpy array
    return np.array(pil_img)

def rotate_image(image, angle):
    """
    Rotate an image by a given angle
    
    Args:
        image (numpy.ndarray): Input image
        angle (float): Rotation angle in degrees
        
    Returns:
        numpy.ndarray: Rotated image
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return rotated_image

def create_mosaic_sample(image_paths, size=(640, 640)):
    """
    Create a mosaic sample from multiple images
    
    Args:
        image_paths (list): List of image paths
        size (tuple): Output size (width, height)
        
    Returns:
        numpy.ndarray: Mosaic image
    """
    if len(image_paths) < 4:
        # If not enough images, duplicate some
        image_paths = image_paths * (4 // len(image_paths) + 1)
    
    # Select 4 random images
    selected_paths = random.sample(image_paths, 4)
    
    # Create mosaic canvas
    mosaic = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Calculate tile size
    tile_width, tile_height = size[0] // 2, size[1] // 2
    
    # Place images in 2x2 grid
    for i, img_path in enumerate(selected_paths[:4]):
        # Read image
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
        else:
            img = cv2.imread(str(img_path))
            
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to tile size
        img = cv2.resize(img, (tile_width, tile_height))
        
        # Calculate position
        row, col = i // 2, i % 2
        y1, y2 = row * tile_height, (row + 1) * tile_height
        x1, x2 = col * tile_width, (col + 1) * tile_width
        
        # Place image
        mosaic[y1:y2, x1:x2] = img
    
    return mosaic

def apply_cutout(image, num_holes=8, max_h_size=64, max_w_size=64):
    """
    Apply cutout augmentation (randomly mask out rectangular regions)
    
    Args:
        image (numpy.ndarray): Input image
        num_holes (int): Number of holes to cut out
        max_h_size (int): Maximum height of each hole
        max_w_size (int): Maximum width of each hole
        
    Returns:
        numpy.ndarray: Image with cutout applied
    """
    height, width = image.shape[:2]
    result = image.copy()
    
    for _ in range(num_holes):
        # Random hole size
        h_size = random.randint(max_h_size // 4, max_h_size)
        w_size = random.randint(max_w_size // 4, max_w_size)
        
        # Random position
        y1 = random.randint(0, height - h_size)
        x1 = random.randint(0, width - w_size)
        
        # Fill with gray (simulate empty shelf)
        result[y1:y1 + h_size, x1:x1 + w_size] = np.array([127, 127, 127])
    
    return result

def color_jitter(image, h_factor=0.1, s_factor=0.7, v_factor=0.4):
    """
    Apply color jitter in HSV space
    
    Args:
        image (numpy.ndarray): Input RGB image
        h_factor (float): Hue variation factor
        s_factor (float): Saturation variation factor
        v_factor (float): Value variation factor
        
    Returns:
        numpy.ndarray: Color jittered image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Random adjustments
    h_gain = random.uniform(1 - h_factor, 1 + h_factor)
    s_gain = random.uniform(1 - s_factor, 1 + s_factor)
    v_gain = random.uniform(1 - v_factor, 1 + v_factor)
    
    # Apply adjustments
    hsv[:, :, 0] = (hsv[:, :, 0] * h_gain) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_gain, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_gain, 0, 255)
    
    # Convert back to RGB
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def add_noise(image, noise_type='gaussian', amount=0.05):
    """
    Add noise to an image
    
    Args:
        image (numpy.ndarray): Input image
        noise_type (str): Type of noise ('gaussian', 'salt_pepper')
        amount (float): Noise amount/strength
        
    Returns:
        numpy.ndarray: Noisy image
    """
    result = image.copy()
    
    if noise_type == 'gaussian':
        # Gaussian noise
        mean = 0
        stddev = amount * 255
        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        s_vs_p = 0.5  # Ratio of salt vs. pepper
        num_pixels = int(amount * image.size)
        
        # Salt (white) mode
        salt_coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
        result[salt_coords[0], salt_coords[1], :] = 255
        
        # Pepper (black) mode
        pepper_coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
        result[pepper_coords[0], pepper_coords[1], :] = 0
    
    return result

def generate_augmentation_examples(data_yaml_path, output_dir):
    """
    Generate examples of advanced augmentation techniques
    
    Args:
        data_yaml_path (str): Path to data.yaml file
        output_dir (str): Directory to save augmentation examples
        
    Returns:
        matplotlib.figure.Figure: Figure with augmentation examples
    """
    # Create output directory for augmented samples
    augmented_dir = os.path.join(output_dir, 'augmented_samples')
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Load data configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get test images path
    base_dir = os.path.dirname(data_yaml_path)
    test_dir = os.path.join(base_dir, data_config['test'].replace('../', ''))
    
    # Get a random image from test set
    test_images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
    if not test_images:
        print("No test images found!")
        return None
    
    sample_image_path = str(random.choice(test_images))
    image = cv2.imread(sample_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define augmentation pipeline
    augmentations = [
        ("Original", lambda img: img),
        ("Horizontal Flip", lambda img: cv2.flip(img, 1)),
        ("Brightness & Contrast", lambda img: adjust_brightness_contrast(img, 1.5, 1.2)),
        ("Rotation", lambda img: rotate_image(img, 15)),
        ("Mosaic", lambda img: create_mosaic_sample(test_images, size=(img.shape[1], img.shape[0]))),
        ("Cutout", lambda img: apply_cutout(img, num_holes=3, max_h_size=img.shape[0]//5, max_w_size=img.shape[1]//5)),
        ("Color Jitter", lambda img: color_jitter(img)),
        ("Blur", lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
        ("Noise", lambda img: add_noise(img))
    ]
    
    # Create a figure to display augmentations
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()
    
    # Apply and display each augmentation
    for i, (aug_name, aug_func) in enumerate(augmentations):
        augmented = aug_func(image.copy())
        axs[i].imshow(augmented)
        axs[i].set_title(aug_name)
        axs[i].axis('off')
        
        # Save augmented sample
        plt.imsave(f"{augmented_dir}/{aug_name.lower().replace(' ', '_')}.jpg", augmented)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentation_examples.png")
    print(f"Advanced augmentation examples saved to {output_dir}/augmentation_examples.png")
    
    return fig

def create_augmented_dataset(data_yaml_path, output_dir, augmentation_factor=0.5):
    """
    Create an augmented dataset with advanced techniques
    
    Args:
        data_yaml_path (str): Path to data.yaml file
        output_dir (str): Directory to save augmented dataset
        augmentation_factor (float): Fraction of original dataset to augment
        
    Returns:
        str: Path to augmented dataset
    """
    print(f"Creating augmented dataset with factor {augmentation_factor}...")
    
    # Load data configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get train images directory
    base_dir = os.path.dirname(data_yaml_path)
    train_dir = os.path.join(base_dir, data_config['train'].replace('../', ''))
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    
    # Create output directories
    augmented_images_dir = os.path.join(output_dir, 'images')
    augmented_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(augmented_labels_dir, exist_ok=True)
    
    # Get list of training images
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Determine how many images to augment
    num_to_augment = int(len(image_files) * augmentation_factor)
    images_to_augment = random.sample(image_files, num_to_augment)
    
    print(f"Augmenting {num_to_augment} images...")
    
    # Augmentation techniques to apply
    augmentation_techniques = [
        ("brightness_contrast", lambda img: adjust_brightness_contrast(img, 
                                                                      brightness=random.uniform(0.7, 1.3), 
                                                                      contrast=random.uniform(0.7, 1.3))),
        ("rotation", lambda img: rotate_image(img, angle=random.uniform(-15, 15))),
        ("color_jitter", lambda img: color_jitter(img)),
        ("noise", lambda img: add_noise(img, amount=random.uniform(0.01, 0.05))),
        ("blur", lambda img: cv2.GaussianBlur(img, (5, 5), random.uniform(0.5, 1.5)))
    ]
    
    # Copy original images and labels first
    for img_file in image_files:
        img_path = os.path.join(train_images_dir, img_file)
        shutil.copy(img_path, os.path.join(augmented_images_dir, img_file))
        
        # Copy corresponding label file if it exists
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(augmented_labels_dir, label_file))
    
    # Apply augmentations
    for img_file in images_to_augment:
        img_path = os.path.join(train_images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_file)
        
        # Skip if label doesn't exist
        if not os.path.exists(label_path):
            continue
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply random augmentations
        for aug_idx, (aug_name, aug_func) in enumerate(random.sample(augmentation_techniques, 2)):
            # Apply augmentation
            augmented = aug_func(image.copy())
            
            # Save augmented image
            aug_img_file = f"{os.path.splitext(img_file)[0]}_{aug_name}_{aug_idx}.jpg"
            aug_img_path = os.path.join(augmented_images_dir, aug_img_file)
            plt.imsave(aug_img_path, augmented)
            
            # Copy label (augmentations preserve bounding boxes)
            aug_label_file = f"{os.path.splitext(img_file)[0]}_{aug_name}_{aug_idx}.txt"
            aug_label_path = os.path.join(augmented_labels_dir, aug_label_file)
            shutil.copy(label_path, aug_label_path)
    
    print(f"Augmented dataset created at {output_dir}")
    return output_dir

# Add missing import for shutil
import shutil

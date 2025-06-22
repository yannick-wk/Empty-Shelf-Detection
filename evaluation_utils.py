#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed Evaluation Utilities for Improved Empty Shelf Detection

This module provides functions for model evaluation, test-time augmentation,
and visualization of predictions. Fixed version that removes the save_txt call
that was causing AttributeError.
"""

import os
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import time

def evaluate_model(model, data_yaml_path, conf_threshold=0.25, iou_threshold=0.65, output_dir=None, split='val'):
    """
    Evaluate a YOLOv8 model on validation or test set
    
    Args:
        model: YOLOv8 model
        data_yaml_path (str): Path to data.yaml file
        conf_threshold (float): Confidence threshold for predictions
        iou_threshold (float): IoU threshold for predictions
        output_dir (str, optional): Directory to save evaluation results
        split (str): Dataset split to evaluate on ('val' or 'test')
        
    Returns:
        dict: Evaluation metrics and results
    """
    print(f"Evaluating model on {split} set...")
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run validation with specified thresholds
    results = model.val(data=data_yaml_path, split=split, conf=conf_threshold, iou=iou_threshold)
    
    # Extract metrics
    metrics = {
        'map50': results.box.map50,
        'map': results.box.map,  # mAP50-95
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16)
    }
    
    # Print metrics
    print(f"\nEvaluation Results ({split} set):")
    print(f"mAP@0.5: {metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Save results if output directory is provided
    if output_dir:
        # Try to save plots if available
        try:
            # Plot confusion matrix if available
            if hasattr(results, 'plot_confusion_matrix'):
                results.plot_confusion_matrix(save_dir=output_dir)
            
            # Save PR curve if available
            if hasattr(results, 'plot_pr_curve'):
                results.plot_pr_curve(save_dir=output_dir)
                
            # Save other plots if available
            if hasattr(model, 'plot_results'):
                model.plot_results(save_dir=output_dir)
        except Exception as e:
            print(f"Could not save evaluation plots: {e}")
    
    return {'metrics': metrics, 'results': results}

def evaluate_with_tta(model, data_yaml_path, conf_threshold=0.25, iou_threshold=0.65, output_dir=None, split='val'):
    """
    Evaluate a YOLOv8 model with test-time augmentation (TTA)
    
    Args:
        model: YOLOv8 model
        data_yaml_path (str): Path to data.yaml file
        conf_threshold (float): Confidence threshold for predictions
        iou_threshold (float): IoU threshold for predictions
        output_dir (str, optional): Directory to save evaluation results
        split (str): Dataset split to evaluate on ('val' or 'test')
        
    Returns:
        dict: Evaluation metrics and results with TTA
    """
    print(f"Evaluating model with test-time augmentation on {split} set...")
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run validation with TTA and specified thresholds
    results = model.val(data=data_yaml_path, split=split, augment=True, conf=conf_threshold, iou=iou_threshold)
    
    # Extract metrics
    metrics = {
        'map50': results.box.map50,
        'map': results.box.map,  # mAP50-95
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16)
    }
    
    # Print metrics
    print(f"\nEvaluation Results with TTA ({split} set):")
    print(f"mAP@0.5: {metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Save results if output directory is provided
    if output_dir:
        # Try to save plots if available
        try:
            # Plot confusion matrix if available
            if hasattr(results, 'plot_confusion_matrix'):
                results.plot_confusion_matrix(save_dir=output_dir)
            
            # Save PR curve if available
            if hasattr(results, 'plot_pr_curve'):
                results.plot_pr_curve(save_dir=output_dir)
                
            # Save other plots if available
            if hasattr(model, 'plot_results'):
                model.plot_results(save_dir=output_dir)
        except Exception as e:
            print(f"Could not save evaluation plots: {e}")
    
    return {'metrics': metrics, 'results': results}

def compare_models(model_metrics_list=None, model_names=None, results_dict=None, output_path=None):
    """
    Compare metrics from multiple models
    
    Args:
        model_metrics_list (list, optional): List of model metrics dictionaries
        model_names (list, optional): List of model names
        results_dict (dict, optional): Dictionary of model metrics with model names as keys
        output_path (str, optional): Path to save the comparison plot
        
    Returns:
        pandas.DataFrame: DataFrame with model comparison
    """
    # Handle both calling conventions
    if results_dict is not None:
        # Use results_dict directly
        comparison_data = {}
        metrics_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
        metrics_keys = ['map50', 'map', 'precision', 'recall', 'f1']
        
        for model_name, metrics in results_dict.items():
            comparison_data[model_name] = [metrics[key] for key in metrics_keys]
    else:
        # Use model_metrics_list and model_names
        comparison_data = {}
        metrics_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
        metrics_keys = ['map50', 'map', 'precision', 'recall', 'f1']
        
        for i, metrics in enumerate(model_metrics_list):
            model_name = model_names[i]
            comparison_data[model_name] = [metrics[key] for key in metrics_keys]
    
    comparison_df = pd.DataFrame(comparison_data, index=metrics_names)
    
    # Print comparison
    print("\nModel Comparison:")
    print(comparison_df.round(4))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Comparison plot saved to: {output_path}")
    
    return comparison_df


def visualize_predictions(image, boxes, confidences, class_names=None, color=(255, 0, 0), thickness=2, font_scale=0.5):
    """
    Visualize model predictions on an image
    
    Args:
        image (numpy.ndarray): Image to visualize predictions on
        boxes (numpy.ndarray): Bounding boxes in format [x1, y1, x2, y2]
        confidences (numpy.ndarray): Confidence scores for each box
        class_names (list, optional): List of class names
        color (tuple): RGB color for bounding boxes
        thickness (int): Line thickness for bounding boxes
        font_scale (float): Font scale for text
        
    Returns:
        numpy.ndarray: Image with visualized predictions
    """
    # Create a copy of the image for drawing
    img_with_boxes = image.copy() if isinstance(image, np.ndarray) else np.array(image).copy()
    
    # Ensure image is RGB if it's grayscale
    if len(img_with_boxes.shape) == 2:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)
    elif img_with_boxes.shape[2] == 4:  # RGBA
        img_with_boxes = img_with_boxes[:, :, :3]  # Convert to RGB
    
    # Draw bounding boxes
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if class_names is not None and i < len(class_names):
            label = f"{class_names[i]}: {conf:.2f}"
        else:
            label = f"Empty Shelf: {conf:.2f}"
        
        # Calculate text size and position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Draw text background
        cv2.rectangle(img_with_boxes, 
                     (x1, y1 - text_size[1] - 5), 
                     (x1 + text_size[0], y1), 
                     color, -1)  # -1 for filled rectangle
        
        # Draw text
        cv2.putText(img_with_boxes, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return img_with_boxes

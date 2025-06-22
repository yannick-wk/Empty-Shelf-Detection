#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Utilities for Improved Empty Shelf Detection

This module provides functions for creating and using model ensembles
to improve detection performance.
"""

import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import yaml

def create_ensemble(model_paths, weights=None):
    """
    Create an ensemble of models
    
    Args:
        model_paths (list): List of paths to model weights
        weights (list, optional): List of weights for each model
        
    Returns:
        list: List of loaded models
    """
    models = []
    
    # Normalize weights if provided
    if weights is not None and len(weights) != len(model_paths):
        raise ValueError("Number of weights must match number of models")
    
    if weights is None:
        weights = [1.0] * len(model_paths)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Load models
    for i, path in enumerate(model_paths):
        try:
            model = YOLO(path)
            models.append(model)
            print(f"Loaded model {i+1}/{len(model_paths)}: {Path(path).name} (weight: {weights[i]:.2f})")
        except Exception as e:
            print(f"Error loading model {path}: {e}")
    
    return models, weights

def ensemble_predict(models, weights, image_path, conf_threshold=0.25, iou_threshold=0.5):
    """
    Make prediction using ensemble of models
    
    Args:
        models (list): List of YOLO models
        weights (list): List of weights for each model
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        ultralytics.engine.results.Results: Ensemble prediction result
    """
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's already a numpy array
        image = image_path
    
    # Get predictions from each model
    all_boxes = []
    all_scores = []
    
    for i, model in enumerate(models):
        results = model.predict(image, conf=conf_threshold, verbose=False)[0]
        
        if len(results.boxes) > 0:
            # Get boxes and scores
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            
            # Apply model weight to scores
            scores = scores * weights[i]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
    
    # If no predictions from any model, return empty result
    if not all_boxes:
        return models[0].predict(image, conf=conf_threshold, verbose=False)[0]
    
    # Combine all predictions
    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    
    # Apply weighted non-maximum suppression
    indices = weighted_nms(boxes, scores, iou_threshold)
    
    # Create result object based on first model's prediction
    result = models[0].predict(image, conf=conf_threshold, verbose=False)[0]
    
    # Update boxes and scores
    if len(indices) > 0:
        result.boxes.xyxy = torch.tensor(boxes[indices], device=result.boxes.xyxy.device)
        result.boxes.conf = torch.tensor(scores[indices], device=result.boxes.conf.device)
        result.boxes.cls = torch.zeros(len(indices), device=result.boxes.cls.device)  # All class 0 (empty shelf)
    else:
        # No boxes after NMS
        result.boxes.xyxy = torch.zeros((0, 4), device=result.boxes.xyxy.device)
        result.boxes.conf = torch.zeros(0, device=result.boxes.conf.device)
        result.boxes.cls = torch.zeros(0, device=result.boxes.cls.device)
    
    return result

def weighted_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply weighted non-maximum suppression
    
    Args:
        boxes (numpy.ndarray): Bounding boxes
        scores (numpy.ndarray): Confidence scores
        iou_threshold (float): IoU threshold
        
    Returns:
        numpy.ndarray: Indices of kept boxes
    """
    # Sort by score
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    
    kept_indices = []
    original_indices = np.arange(len(boxes))[order]
    
    while len(boxes) > 0:
        # Keep the box with highest score
        kept_indices.append(original_indices[0])
        
        if len(boxes) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = calculate_ious(boxes[0], boxes[1:])
        
        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        original_indices = original_indices[1:][mask]
    
    return kept_indices

def calculate_ious(box, boxes):
    """
    Calculate IoU between a box and an array of boxes
    
    Args:
        box (numpy.ndarray): Single box [x1, y1, x2, y2]
        boxes (numpy.ndarray): Array of boxes
        
    Returns:
        numpy.ndarray: IoU values
    """
    # Calculate intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    # Calculate area of intersection
    width = np.maximum(0, x2 - x1)
    height = np.maximum(0, y2 - y1)
    intersection = width * height
    
    # Calculate area of box and boxes
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Calculate union
    union = box_area + boxes_area - intersection
    
    # Calculate IoU
    iou = intersection / union
    
    return iou

def ensemble_predict_with_tta(models, weights, image_path, conf_threshold=0.25, iou_threshold=0.5):
    """
    Make prediction using ensemble of models with test-time augmentation
    
    Args:
        models (list): List of YOLO models
        weights (list): List of weights for each model
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        ultralytics.engine.results.Results: Ensemble prediction result
    """
    # Get predictions from each model with TTA
    all_boxes = []
    all_scores = []
    
    for i, model in enumerate(models):
        results = model.predict(image_path, conf=conf_threshold, augment=True, verbose=False)[0]
        
        if len(results.boxes) > 0:
            # Get boxes and scores
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            
            # Apply model weight to scores
            scores = scores * weights[i]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
    
    # If no predictions from any model, return empty result
    if not all_boxes:
        return models[0].predict(image_path, conf=conf_threshold, augment=True, verbose=False)[0]
    
    # Combine all predictions
    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    
    # Apply weighted non-maximum suppression
    indices = weighted_nms(boxes, scores, iou_threshold)
    
    # Create result object based on first model's prediction
    result = models[0].predict(image_path, conf=conf_threshold, augment=True, verbose=False)[0]
    
    # Update boxes and scores
    if len(indices) > 0:
        result.boxes.xyxy = torch.tensor(boxes[indices], device=result.boxes.xyxy.device)
        result.boxes.conf = torch.tensor(scores[indices], device=result.boxes.conf.device)
        result.boxes.cls = torch.zeros(len(indices), device=result.boxes.cls.device)  # All class 0 (empty shelf)
    else:
        # No boxes after NMS
        result.boxes.xyxy = torch.zeros((0, 4), device=result.boxes.xyxy.device)
        result.boxes.conf = torch.zeros(0, device=result.boxes.conf.device)
        result.boxes.cls = torch.zeros(0, device=result.boxes.cls.device)
    
    return result

def evaluate_ensemble(models, weights, data_yaml_path, split='val', conf_threshold=0.25, iou_threshold=0.5):
    """
    Evaluate ensemble on validation or test set
    
    Args:
        models (list): List of YOLO models
        weights (list): List of weights for each model
        data_yaml_path (str): Path to data.yaml file
        split (str): Dataset split to evaluate on ('val' or 'test')
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"Evaluating ensemble on {split} set...")
    
    # Load data configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get dataset path
    base_dir = os.path.dirname(data_yaml_path)
    dataset_dir = os.path.join(base_dir, data_config[split].replace('../', ''))
    
    # Get image and label paths
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    image_paths = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    
    # Initialize metrics
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    # Process each image
    for img_path in image_paths:
        # Get ground truth boxes
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        gt_boxes = []
        
        if label_path.exists():
            # Load image to get dimensions
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2]
            
            # Read label file
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height
                        cls, x_center, y_center, width, height = map(float, parts[:5])
                        
                        # Convert to absolute coordinates (xyxy format)
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height
                        
                        gt_boxes.append([x1, y1, x2, y2])
        
        # Get ensemble predictions
        result = ensemble_predict(models, weights, str(img_path), conf_threshold, iou_threshold)
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        
        # Match predictions to ground truth
        matched_gt = [False] * len(gt_boxes)
        
        for pred_box in pred_boxes:
            # Check if prediction matches any ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i]:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(pred_box, np.array(gt_box))
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # If IoU > threshold, it's a true positive
            if best_iou >= 0.5:  # IoU threshold for TP
                tp += 1
                matched_gt[best_gt_idx] = True
            else:
                fp += 1
        
        # Count unmatched ground truth boxes as false negatives
        fn += sum(not matched for matched in matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Approximate mAP
    mAP50 = precision * recall  # Simple approximation
    
    metrics = {
        'mAP50': mAP50,
        'mAP50-95': mAP50 * 0.7,  # Approximation
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    
    # Print metrics
    print(f"\nEnsemble Evaluation Results ({split} set):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"mAP@0.5 (approx): {mAP50:.4f}")
    
    return metrics

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    
    Args:
        box1 (numpy.ndarray): First box [x1, y1, x2, y2]
        box2 (numpy.ndarray): Second box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def visualize_ensemble_predictions(models, weights, image_path, conf_threshold=0.25, output_path=None):
    """
    Visualize ensemble predictions on an image
    
    Args:
        models (list): List of YOLO models
        weights (list): List of weights for each model
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold
        output_path (str, optional): Path to save visualization
        
    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get ensemble prediction
    result = ensemble_predict(models, weights, image_path, conf_threshold)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Get boxes and scores
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    
    # Draw boxes
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        
        # Create rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor='purple', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add confidence text
        plt.text(x1, y1-10, f"Empty Shelf: {score:.2f}", 
                color='white', fontsize=12, 
                bbox=dict(facecolor='purple', alpha=0.5))
    
    plt.axis('off')
    plt.title(f"Ensemble Detection - {Path(image_path).name}")
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.tight_layout()
    
    return plt.gcf()

def compare_models_with_ensemble(models, weights, image_path, conf_threshold=0.25, output_path=None):
    """
    Compare individual model predictions with ensemble prediction
    
    Args:
        models (list): List of YOLO models
        weights (list): List of weights for each model
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold
        output_path (str, optional): Path to save visualization
        
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    n_cols = len(models) + 1  # +1 for ensemble
    fig, axs = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Plot individual model predictions
    for i, model in enumerate(models):
        # Get prediction
        result = model.predict(image_path, conf=conf_threshold, verbose=False)[0]
        
        # Plot image
        axs[i].imshow(image)
        axs[i].set_title(f"Model {i+1} (w={weights[i]:.2f})")
        axs[i].axis('off')
        
        # Get boxes and scores
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        # Draw boxes
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            
            # Create rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='red', linewidth=2)
            axs[i].add_patch(rect)
            
            # Add confidence text
            axs[i].text(x1, y1-5, f"{score:.2f}", 
                       color='white', fontsize=10, 
                       bbox=dict(facecolor='red', alpha=0.5))
    
    # Plot ensemble prediction
    result = ensemble_predict(models, weights, image_path, conf_threshold)
    
    # Plot image
    axs[-1].imshow(image)
    axs[-1].set_title("Ensemble")
    axs[-1].axis('off')
    
    # Get boxes and scores
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    
    # Draw boxes
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        
        # Create rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor='purple', linewidth=2)
        axs[-1].add_patch(rect)
        
        # Add confidence text
        axs[-1].text(x1, y1-5, f"{score:.2f}", 
                   color='white', fontsize=10, 
                   bbox=dict(facecolor='purple', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Comparison visualization saved to {output_path}")
    
    return fig

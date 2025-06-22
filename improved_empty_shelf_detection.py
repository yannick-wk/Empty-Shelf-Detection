#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Empty Shelf Detection - Main Script

This script implements an improved version of the empty shelf detection model with:
1. Larger base model (YOLOv8m instead of YOLOv8n)
2. Advanced data augmentation
3. Hyperparameter optimization
4. Test-time augmentation
5. Ensemble methods
6. Domain adaptation techniques
7. Comprehensive evaluation framework
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
from pathlib import Path
import random
import time

# Import utility modules
from model_utils import load_model, create_improved_model, train_improved_model, plot_training_metrics, display_metrics_summary
from augmentation_utils import generate_augmentation_examples, create_augmented_dataset
from evaluation_utils import evaluate_model, evaluate_with_tta, compare_models, visualize_predictions, visualize_batch_predictions
from ensemble_utils import create_ensemble, ensemble_predict, evaluate_ensemble, visualize_ensemble_predictions, compare_models_with_ensemble

# Set paths
BASE_DIR = "/home/yannick/tf-nn/PERSONAL"
DATA_YAML_PATH = f"{BASE_DIR}/data/empty_shelves/data.yaml"
ORIGINAL_MODEL_PATH = f"{BASE_DIR}/runs/detect/yolov8n_empty_shelf_detector_v1/weights/best.pt"
RESULTS_CSV = f"{BASE_DIR}/runs/detect/yolov8n_empty_shelf_detector_v1/results.csv"
OUTPUT_DIR = f"{BASE_DIR}/improved_model_evaluation"
IMPROVED_MODEL_NAME = "yolov8m_improved_empty_shelf_detector"
IMPROVED_MODEL_DIR = f"{BASE_DIR}/runs/detect/{IMPROVED_MODEL_NAME}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration settings
CONFIG = {
    'base_model': 'yolov8m.pt',  # Upgraded from yolov8n to yolov8m
    'epochs': 100,               # Increased from 50 to 100
    'batch_size': 16,            # Adjusted batch size
    'image_size': 832,           # Increased from 640 to 832
    'optimizer': 'AdamW',        # Changed from auto to AdamW
    'initial_lr': 0.001,         # Custom learning rate
    'final_lr': 0.0001,          # Learning rate at end of training
    'weight_decay': 0.0005,      # L2 regularization
    'conf_threshold': 0.25,      # Default confidence threshold
    'iou_threshold': 0.65,       # Increased from 0.7 to 0.65 for better recall
    'use_tta': True,             # Enable test-time augmentation
    'ensemble_models': True,     # Use model ensemble
    'domain_adaptation': True,   # Apply domain adaptation techniques
    'augmentation_strength': 'strong',  # Level of data augmentation
    'model_name': IMPROVED_MODEL_NAME,
}

def main():
    """Main function to run the improved empty shelf detection workflow"""
    print("=" * 70)
    print("IMPROVED EMPTY SHELF DETECTION")
    print("=" * 70)
    
    # Step 1: Load original model for comparison
    print("\n1. Loading original model for baseline comparison...")
    original_model = load_model(ORIGINAL_MODEL_PATH)
    
    # Step 2: Generate augmentation examples
    print("\n2. Generating advanced augmentation examples...")
    augmentation_fig = generate_augmentation_examples(DATA_YAML_PATH, OUTPUT_DIR)
    
    # Step 3: Train improved model with optimized hyperparameters
    print("\n3. Training improved model with optimized hyperparameters...")
    improved_model_path = train_improved_model(
        CONFIG['base_model'],
        DATA_YAML_PATH, 
        CONFIG['model_name'],
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size'],
        optimizer=CONFIG['optimizer'],
        initial_lr=CONFIG['initial_lr'],
        final_lr=CONFIG['final_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Step 4: Load improved model
    print("\n4. Loading improved model...")
    improved_model = load_model(improved_model_path)
    
    # Step 5: Evaluate and compare models
    print("\n5. Evaluating and comparing models...")
    # Evaluate original model
    original_metrics = evaluate_model(original_model, DATA_YAML_PATH, split="val")
    
    # Evaluate improved model
    improved_metrics = evaluate_model(improved_model, DATA_YAML_PATH, split="val")
    
    # Compare models
    comparison_results = compare_models(
        [original_metrics, improved_metrics],
        ["Original Model (YOLOv8n)", "Improved Model (YOLOv8m)"],
    )
    comparison_results.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")
    print(f"Model comparison saved to: {OUTPUT_DIR}/model_comparison.csv")
    
    # Step 6: Evaluate with test-time augmentation
    print("\n6. Evaluating with test-time augmentation...")
    tta_metrics = evaluate_with_tta(improved_model, DATA_YAML_PATH, split="val")
    
    # Add TTA to comparison
    comparison_results = compare_models(
        [original_metrics, improved_metrics, tta_metrics],
        ["Original Model (YOLOv8n)", "Improved Model (YOLOv8m)", "Improved Model + TTA"],
    )
    comparison_results.to_csv(f"{OUTPUT_DIR}/model_comparison_with_tta.csv")
    print(f"Updated model comparison saved to: {OUTPUT_DIR}/model_comparison_with_tta.csv")
    
    # Step 7: Create and evaluate ensemble
    print("\n7. Creating and evaluating model ensemble...")
    # Create ensemble with original and improved models
    model_paths = [ORIGINAL_MODEL_PATH, improved_model_path]
    
    # Add weights (give more weight to improved model)
    weights = [1.0, 2.0]
    
    # Create ensemble
    ensemble_models, ensemble_weights = create_ensemble(model_paths, weights)
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_ensemble(ensemble_models, ensemble_weights, DATA_YAML_PATH, split="val")
    
    # Add ensemble to comparison
    comparison_results = compare_models(
        [original_metrics, improved_metrics, tta_metrics, ensemble_metrics],
        ["Original Model (YOLOv8n)", "Improved Model (YOLOv8m)", "Improved Model + TTA", "Ensemble Model"],
    )
    comparison_results.to_csv(f"{OUTPUT_DIR}/model_comparison_with_ensemble.csv")
    print(f"Final model comparison saved to: {OUTPUT_DIR}/model_comparison_with_ensemble.csv")
    
    # Step 8: Visualize predictions on test images
    print("\n8. Visualizing predictions on test images...")
    # Get test images from data.yaml
    with open(DATA_YAML_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    
    base_dir = os.path.dirname(DATA_YAML_PATH)
    test_dir = os.path.join(base_dir, data_config['test'].replace('../', ''))
    test_images_dir = os.path.join(test_dir, 'images')
    
    # Create visualizations directory
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get list of test images
    test_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        test_images.extend(list(Path(test_images_dir).glob(ext)))
    test_images = [str(p) for p in test_images]
    
    if test_images:
        # Sample a few images for visualization
        for i, img_path in enumerate(random.sample(test_images, min(5, len(test_images)))):
            # Visualize original model predictions
            output_path = os.path.join(vis_dir, f"original_prediction_{i+1}.jpg")
            visualize_predictions(original_model, img_path, conf_threshold=CONFIG['conf_threshold'], output_path=output_path)
            
            # Visualize improved model predictions
            output_path = os.path.join(vis_dir, f"improved_prediction_{i+1}.jpg")
            visualize_predictions(improved_model, img_path, conf_threshold=CONFIG['conf_threshold'], output_path=output_path)
            
            # Visualize improved model with TTA
            output_path = os.path.join(vis_dir, f"improved_tta_prediction_{i+1}.jpg")
            visualize_predictions(improved_model, img_path, conf_threshold=CONFIG['conf_threshold'], output_path=output_path, tta=True)
            
            # Visualize ensemble predictions
            if 'ensemble_models' in locals() and 'ensemble_weights' in locals():
                output_path = os.path.join(vis_dir, f"ensemble_prediction_{i+1}.jpg")
                visualize_ensemble_predictions(ensemble_models, ensemble_weights, img_path, conf_threshold=CONFIG['conf_threshold'], output_path=output_path)
                
                # Compare all models with ensemble
                output_path = os.path.join(vis_dir, f"model_comparison_{i+1}.jpg")
                compare_models_with_ensemble(ensemble_models, ensemble_weights, img_path, conf_threshold=CONFIG['conf_threshold'], output_path=output_path)
        
        # Batch visualization
        batch_vis_dir = os.path.join(OUTPUT_DIR, "batch_visualizations")
        os.makedirs(batch_vis_dir, exist_ok=True)
        visualize_batch_predictions(improved_model, test_images_dir, batch_vis_dir, conf_threshold=CONFIG['conf_threshold'], max_images=10)
    
    print("\n" + "=" * 70)
    print("IMPROVED EMPTY SHELF DETECTION COMPLETED")
    print("=" * 70)
    print(f"All results saved to: {OUTPUT_DIR}")

def run_domain_adaptation(model, target_images_dir, data_yaml_path, output_dir):
    """Run domain adaptation fine-tuning if enabled in CONFIG"""
    if not CONFIG['domain_adaptation'] or not os.path.exists(target_images_dir):
        return model
    
    print("\nPerforming domain adaptation fine-tuning...")
    from model_utils import fine_tune_for_domain_adaptation
    
    # Fine-tune the model on target domain images
    adapted_model_path = fine_tune_for_domain_adaptation(
        model.ckpt_path,
        target_images_dir,
        data_yaml_path,
        f"{CONFIG['model_name']}_domain_adapted",
        epochs=10,  # Short fine-tuning
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size']
    )
    
    # Load the domain-adapted model
    adapted_model = load_model(adapted_model_path)
    
    # Evaluate the domain-adapted model
    adapted_metrics = evaluate_model(adapted_model, data_yaml_path, split="val")
    
    # Save evaluation results
    pd.DataFrame({
        "Metric": list(adapted_metrics.keys()),
        "Value": list(adapted_metrics.values())
    }).to_csv(f"{output_dir}/domain_adaptation_results.csv", index=False)
    
    print(f"Domain adaptation results saved to: {output_dir}/domain_adaptation_results.csv")
    return adapted_model

def plot_boxes(ax, result, title):
    """Plot bounding boxes on the given axis"""
    if hasattr(result, 'boxes'):
        # For YOLOv8 result objects
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-10, f'Empty Shelf: {conf:.2f}', color='red', fontsize=10, backgroundcolor='white')
    elif isinstance(result, dict) and 'boxes' in result:
        # For ensemble result dictionary
        for box in result['boxes']:
            x1, y1, x2, y2 = box[:4].astype(int)
            conf = box[4]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1-10, f'Empty Shelf: {conf:.2f}', color='red', fontsize=10, backgroundcolor='white')
    
    ax.set_title(title)
    ax.axis('off')

def visualize_comparison(image_path, original_result, improved_result, ensemble_result, output_path):
    """Visualize and compare detection results from different models"""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3 if ensemble_result else 2, figsize=(18, 6))
    
    # Original model result
    axes[0].imshow(img)
    plot_boxes(axes[0], original_result, "Original Model (YOLOv8n)")
    
    # Improved model result
    axes[1].imshow(img)
    plot_boxes(axes[1], improved_result, "Improved Model (YOLOv8m)")
    
    # Ensemble result (if available)
    if ensemble_result:
        axes[2].imshow(img)
        plot_boxes(axes[2], ensemble_result, "Ensemble Model")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Comparison visualization saved to: {output_path}")

def plot_boxes(ax, result, title):
    """Plot bounding boxes on axis"""
    ax.set_title(title)
    
    # If no boxes, just display the image
    if result is None or len(result.boxes) == 0:
        ax.set_xlabel("No detections")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Get boxes, confidence scores and class IDs
    boxes = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    
    # Plot each box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            fill=False, 
            edgecolor='red', 
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1-5, 
            f"Empty Shelf: {conf[i]:.2f}", 
            color='red', 
            fontsize=10, 
            backgroundcolor='white'
        )
    
    # Add inference time if available
    if hasattr(result, 'speed'):
        inference_time = result.speed.get('inference', None)
        if inference_time:
            ax.set_xlabel(f"Inference: {inference_time:.1f}ms")
    
    ax.set_xticks([])
    ax.set_yticks([])

if __name__ == "__main__":
    main()

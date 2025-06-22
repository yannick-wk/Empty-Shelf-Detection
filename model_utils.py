#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Utilities for Improved Empty Shelf Detection

This module provides functions for loading, creating, and training improved models
for empty shelf detection.
"""

import os
import time
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_model(model_path):
    """Load a YOLOv8 model from the specified path"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = YOLO(model_path)
    print(f"Model loaded from: {model_path}")
    return model

def create_improved_model(base_model='yolov8m.pt'):
    """Create an improved YOLOv8 model for empty shelf detection"""
    model = YOLO(base_model)
    print(f"Created improved model based on {base_model}")
    return model

def train_improved_model_fixed(base_model, data_yaml_path, model_name, 
                         epochs=100, batch_size=16, image_size=832,
                         optimizer='AdamW', initial_lr=0.001, final_lr=0.0001,
                         weight_decay=0.0005):
    """
    Train an improved YOLOv8 model with optimized hyperparameters
    
    Args:
        base_model (str or YOLO): Path to base model, model name, or YOLO model object
        data_yaml_path (str): Path to data.yaml file
        model_name (str): Name for the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        image_size (int): Input image size
        optimizer (str): Optimizer to use ('AdamW', 'SGD', etc.)
        initial_lr (float): Initial learning rate
        final_lr (float): Final learning rate (for cosine scheduler)
        weight_decay (float): Weight decay for regularization
        
    Returns:
        str: Path to the best trained model weights
    """
    import time
    from pathlib import Path
    from ultralytics import YOLO
    
    start_time = time.time()
    
    # Create model - handle both string paths and YOLO objects
    if isinstance(base_model, str):
        model = YOLO(base_model)
    else:
        # If it's already a YOLO model, use it directly
        model = base_model
    
    # Train with improved parameters
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        name=model_name,
        patience=20,             # Early stopping patience
        optimizer=optimizer,
        lr0=initial_lr,          # Initial learning rate
        lrf=final_lr,            # Final learning rate
        momentum=0.937,          # SGD momentum/Adam beta1
        weight_decay=weight_decay, # Weight decay
        warmup_epochs=3.0,       # Warmup epochs
        warmup_momentum=0.8,     # Warmup initial momentum
        warmup_bias_lr=0.1,      # Warmup initial bias lr
        box=0.05,                # Box loss gain
        cls=0.3,                 # Class loss gain
        hsv_h=0.015,             # Hue augmentation
        hsv_s=0.7,               # Saturation augmentation
        hsv_v=0.4,               # Value (brightness) augmentation
        degrees=10.0,            # Rotation degrees
        translate=0.1,           # Translation
        scale=0.5,               # Scale
        shear=2.0,               # Shear
        perspective=0.0001,      # Perspective
        flipud=0.0,              # Flip up-down
        fliplr=0.5,              # Flip left-right
        mosaic=1.0,              # Mosaic probability
        mixup=0.15,              # MixUp probability
        copy_paste=0.1,          # Copy-paste probability
        augment=True,            # Use augmentation
        cos_lr=True,             # Use cosine learning rate scheduler
        close_mosaic=10,         # Disable mosaic in last 10 epochs
        verbose=True,
        device='',               # Auto-select device
        save=True,               # Save checkpoints
        save_period=10,          # Save checkpoint every 10 epochs
        plots=True,              # Generate plots
        exist_ok=True            # Overwrite existing experiment
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Get path to best model
    best_model_path = str(Path(model.trainer.best).resolve())
    print(f"Best model saved to: {best_model_path}")
    
    return best_model_path

def plot_training_metrics(results_csv, output_dir=None):
    """
    Plot training metrics from results.csv
    
    Args:
        results_csv (str): Path to results.csv file
        output_dir (str, optional): Directory to save plots
        
    Returns:
        matplotlib.figure.Figure: Figure with training metrics plots
    """
    # Load results
    results = pd.read_csv(results_csv)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot training losses
    axs[0, 0].plot(results['epoch'], results['train/box_loss'], label='Box Loss')
    axs[0, 0].plot(results['epoch'], results['train/cls_loss'], label='Class Loss')
    axs[0, 0].plot(results['epoch'], results['train/dfl_loss'], label='DFL Loss')
    axs[0, 0].set_title('Training Losses')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot validation losses
    axs[0, 1].plot(results['epoch'], results['val/box_loss'], label='Box Loss')
    axs[0, 1].plot(results['epoch'], results['val/cls_loss'], label='Class Loss')
    axs[0, 1].plot(results['epoch'], results['val/dfl_loss'], label='DFL Loss')
    axs[0, 1].set_title('Validation Losses')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot precision, recall
    axs[0, 2].plot(results['epoch'], results['metrics/precision(B)'], label='Precision')
    axs[0, 2].plot(results['epoch'], results['metrics/recall(B)'], label='Recall')
    axs[0, 2].set_title('Precision and Recall')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Value')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # Plot mAP
    axs[1, 0].plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP@0.5')
    axs[1, 0].plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    axs[1, 0].set_title('Mean Average Precision (mAP)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('mAP')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot learning rate
    axs[1, 1].plot(results['epoch'], results['lr/pg0'])
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].grid(True)
    
    # Plot training time
    cumulative_time = results['time'].cumsum()
    axs[1, 2].plot(results['epoch'], cumulative_time)
    axs[1, 2].set_title('Cumulative Training Time')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Time (seconds)')
    axs[1, 2].grid(True)
    
    plt.tight_layout()
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/training_metrics.png")
        print(f"Training metrics plot saved to: {output_dir}/training_metrics.png")
    
    return fig

def display_metrics_summary(results_csv):
    """
    Display a summary of the model metrics from results.csv
    
    Args:
        results_csv (str): Path to results.csv file
        
    Returns:
        pandas.DataFrame: DataFrame with metrics summary
    """
    results = pd.read_csv(results_csv)
    
    # Get the best epoch based on mAP50-95
    best_epoch = results['metrics/mAP50-95(B)'].idxmax() + 1
    best_results = results.iloc[best_epoch-1]
    
    # Get the final epoch results
    final_results = results.iloc[-1]
    
    # Print summary
    print(f"\nTraining completed for {len(results)} epochs")
    print(f"\nBest Model Metrics (Epoch {best_epoch}):\n")
    print(f"mAP@0.5: {best_results['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95: {best_results['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {best_results['metrics/precision(B)']:.4f}")
    print(f"Recall: {best_results['metrics/recall(B)']:.4f}")
    
    print(f"\nFinal Model Metrics (Epoch {len(results)}):\n")
    print(f"mAP@0.5: {final_results['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95: {final_results['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {final_results['metrics/precision(B)']:.4f}")
    print(f"Recall: {final_results['metrics/recall(B)']:.4f}")
    
    # Create DataFrame for comparison
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    best_values = [
        best_results['metrics/mAP50(B)'], 
        best_results['metrics/mAP50-95(B)'],
        best_results['metrics/precision(B)'],
        best_results['metrics/recall(B)']
    ]
    final_values = [
        final_results['metrics/mAP50(B)'], 
        final_results['metrics/mAP50-95(B)'],
        final_results['metrics/precision(B)'],
        final_results['metrics/recall(B)']
    ]
    
    df = pd.DataFrame({
        'Metric': metrics,
        f'Best Model (Epoch {best_epoch})': best_values,
        f'Final Model (Epoch {len(results)})': final_values
    })
    
    return df

def fine_tune_for_domain_adaptation(model, target_images_dir, data_yaml_path, 
                                   epochs=10, batch_size=8, output_name="domain_adapted_model"):
    """
    Fine-tune a model for domain adaptation to a specific environment
    
    Args:
        model: YOLOv8 model to fine-tune
        target_images_dir (str): Directory with images from target domain
        data_yaml_path (str): Path to data.yaml file
        epochs (int): Number of fine-tuning epochs
        batch_size (int): Batch size for fine-tuning
        output_name (str): Name for the fine-tuned model
        
    Returns:
        str: Path to the fine-tuned model
    """
    print(f"Fine-tuning model for domain adaptation with {epochs} epochs")
    
    # Use a lower learning rate for fine-tuning
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=model.model.args['imgsz'] if hasattr(model.model, 'args') else 832,
        name=output_name,
        lr0=0.0001,  # Lower learning rate for fine-tuning
        lrf=0.00001,  # Final learning rate
        optimizer='AdamW',
        patience=5,  # Early stopping patience
        cos_lr=True,
        augment=True,
        exist_ok=True
    )
    
    # Get path to best fine-tuned model
    best_model_path = str(Path(model.trainer.best).resolve())
    print(f"Domain-adapted model saved to: {best_model_path}")
    
    return best_model_path

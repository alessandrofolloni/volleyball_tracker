from pathlib import Path
from ultralytics import YOLO
import wandb
import pandas as pd
import yaml
import random
import shutil
import os

def save_metrics_to_file(metrics, save_dir, experiment, avg_train_loss, avg_val_loss):
    """
    Saves the evaluation metrics to a text file.

    Args:
        metrics (DetMetrics): Metrics object returned by model.val().
        save_dir (str or Path): Directory where the metrics file will be saved.
        experiment (str): Experiment name.
        avg_train_loss (float): Average training loss over all epochs.
        avg_val_loss (float): Average validation loss over all epochs.
    """
    name = f'metrics_report_{experiment}.txt'
    metrics_file = Path(save_dir) / name
    with open(metrics_file, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")

        # Overall Metrics
        f.write(f"Metrics for the model:\n")
        f.write(f"  - Model name: {experiment}\n\n")
        f.write("Overall Performance:\n")

        # Access metrics via attributes
        f.write(f"  - Precision (P): {metrics.box.mp:.4f}\n")
        f.write(f"  - Recall (R): {metrics.box.mr:.4f}\n")
        f.write(f"  - Mean Average Precision @ IoU=0.5 (mAP@0.5): {metrics.box.map50:.4f}\n")
        f.write(f"  - Mean Average Precision @ IoU=0.5:0.95 (mAP@0.5:0.95): {metrics.box.map:.4f}\n\n")

        # Losses
        f.write("Losses:\n")
        if avg_train_loss is not None:
            f.write(f"  - Average Training Loss: {avg_train_loss:.4f}\n")
        else:
            f.write("  - Average Training Loss: N/A\n")

        if avg_val_loss is not None:
            f.write(f"  - Average Validation Loss: {avg_val_loss:.4f}\n")
        else:
            f.write("  - Average Validation Loss: N/A\n")
        f.write("\n")

        # Per-Class Metrics
        f.write("Per-Class Performance:\n")
        if metrics.box.maps is not None and hasattr(metrics.box, 'ap_class_index'):
            # Get class indices and names
            ap_class_indices = metrics.box.ap_class_index  # Indices of classes
            for i, class_idx in enumerate(ap_class_indices):
                class_name = metrics.names[class_idx]
                # Get per-class precision, recall, map50, map
                p, r, ap50, ap = metrics.box.class_result(i)
                f.write(f"Class '{class_name}':\n")
                f.write(f"  - Precision: {p:.4f}\n")
                f.write(f"  - Recall: {r:.4f}\n")
                f.write(f"  - mAP@0.5: {ap50:.4f}\n")
                f.write(f"  - mAP@0.5:0.95: {ap:.4f}\n\n")
        else:
            f.write("Per-class metrics are not available.\n")

    print(f"Metrics report saved to {metrics_file}")

def train_and_evaluate(config):
    """
    Trains YOLOv8 on a custom dataset and evaluates on the test set.

    Args:
        config (dict): Dictionary containing all configuration parameters.
    """
    # Extract configuration parameters
    data_yaml = config['data_yaml']
    data_fraction = config.get('data_fraction', 1.0)  # Default to 100% if not specified

    # Other configuration parameters
    model_name = config['model_name']
    epochs = config['epochs']
    imgsz = config['imgsz']
    project_name = config['project_name']
    batch_size = config['batch_size']
    optimizer = config['optimizer']
    lr = config['learning_rate']
    momentum = config.get('momentum', None)
    weight_decay = config.get('weight_decay', None)
    augment = config['augment']
    verbose = config['verbose']
    save_dir = config.get('save_dir', project_name)

    # Define the experiment name based on parameters
    experiment = f"train_epochs{epochs}_model{Path(model_name).stem}_bs{batch_size}_opt{optimizer}_data{int(data_fraction * 100)}"

    # Initialize Weights & Biases (wandb) if not already initialized
    if wandb.run is None:
        wandb.init(
            project=project_name,
            name=experiment,
            config=config,
            resume='allow'
        )

    # Load the YOLO model
    model = YOLO(model_name)

    # Prepare training parameters
    train_params = {
        'data': data_yaml,  # Will be updated if data_fraction < 1.0
        'epochs': epochs,
        'imgsz': imgsz,
        'val': True,
        'project': project_name,
        'name': experiment,
        'exist_ok': True,
        'verbose': verbose,
        'augment': augment,
        'batch': batch_size,
        'optimizer': optimizer,
        'save_period': 1,  # Save model after every epoch
        'cache': False,    # Set to True if you want to cache images
    }

    # Optionally add optimizer-specific parameters
    if optimizer.lower() == 'adam':
        train_params['lr0'] = lr  # Initial learning rate for Adam
    elif optimizer.lower() == 'sgd':
        train_params['lr0'] = lr
        if momentum is not None:
            train_params['momentum'] = momentum
    elif optimizer.lower() == 'adamw':
        train_params['lr0'] = lr
        if weight_decay is not None:
            train_params['weight_decay'] = weight_decay

    # If data_fraction < 1.0, create a subset of the dataset
    if data_fraction < 1.0:
        # Load the original data.yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)

        # Prepare temporary directories for the subset data
        subset_data_dir = Path(save_dir) / f"subset_data_{int(data_fraction * 100)}"
        subset_data_dir.mkdir(parents=True, exist_ok=True)

        # Function to create subsets for train, valid, test
        def create_data_subset(split_name):
            if split_name not in data:
                return None

            # Get original images directory
            original_images_path = data[split_name]
            if not Path(original_images_path).is_absolute():
                base_path = Path(data.get('path', '.'))
                original_images_path = base_path / original_images_path
            else:
                base_path = Path('/')

            # Get corresponding labels directory
            if split_name == 'val':
                labels_split_name = 'valid'
            else:
                labels_split_name = split_name

            original_labels_path = base_path / labels_split_name / 'labels'

            # Get all image paths
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(original_images_path.glob(f'{ext}'))

            # Subsample the image paths
            subset_size = max(1, int(len(image_paths) * data_fraction))
            subset_image_paths = random.sample(image_paths, subset_size)

            # Create new directories for the subset
            subset_images_dir = subset_data_dir / labels_split_name / 'images'
            subset_labels_dir = subset_data_dir / labels_split_name / 'labels'
            subset_images_dir.mkdir(parents=True, exist_ok=True)
            subset_labels_dir.mkdir(parents=True, exist_ok=True)

            # Copy images and labels to the subset directories
            for image_path in subset_image_paths:
                # Copy image
                dest_image_path = subset_images_dir / image_path.name
                shutil.copy(image_path, dest_image_path)

                # Copy corresponding label file
                label_filename = image_path.stem + '.txt'
                label_path = original_labels_path / label_filename
                if label_path.exists():
                    dest_label_path = subset_labels_dir / label_filename
                    shutil.copy(label_path, dest_label_path)
                else:
                    print(f"Label file not found for image: {image_path}")

            # Return the relative path to the subset images directory
            return str((labels_split_name + '/images'))

        # Create subsets
        subset_train_path = create_data_subset('train')
        subset_val_path = create_data_subset('val')
        subset_test_path = create_data_subset('test')

        # Create a new data.yaml for the subset
        subset_data_yaml = subset_data_dir / 'data.yaml'
        subset_data = {
            'path': str(subset_data_dir),
            'train': subset_train_path if subset_train_path else '',
            'val': subset_val_path if subset_val_path else '',
            'test': subset_test_path if subset_test_path else '',
            'nc': data['nc'] if 'nc' in data else len(data['names']),
            'names': data['names']
        }
        with open(subset_data_yaml, 'w') as f:
            yaml.dump(subset_data, f)

        # Update train_params to use the new data.yaml
        train_params['data'] = str(subset_data_yaml)

        # Also update data_yaml variable for validation step
        data_yaml = str(subset_data_yaml)

    # Proceed with training
    results = model.train(**train_params)

    # Calculate average training and validation losses
    # Access results from the 'metrics' attribute
    training_results = results.metrics

    avg_train_loss = training_results.box_loss
    avg_val_loss = training_results.val_loss

    # Print the loss metrics
    print("Average Losses:")
    print(f"  - Training Loss: {avg_train_loss:.4f}")
    print(f"  - Validation Loss: {avg_val_loss:.4f}")

    # Evaluate the model on the test set
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=imgsz,
        save=True,
        save_txt=True,
        save_json=True,
        plots=True,
        verbose=verbose,
    )

    # Save metrics to a text file, including loss metrics
    save_metrics_to_file(metrics, save_dir, experiment, avg_train_loss, avg_val_loss)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    # ==============================
    #        Configuration
    # ==============================

    config = {
        # === Data Configuration ===
        'data_yaml': "/public.hpc/alessandro.folloni2/volleyball_tracker/data.yaml",
        'data_fraction': 1,

        # === Model Configuration ===
        'model_name': "yolov8s.pt",  # Pre-trained YOLOv8 model

        # === Training Parameters ===
        'epochs': 100,               # Number of training epochs
        'imgsz': 640,                # Image size for training and evaluation (e.g., 640)
        'batch_size': 16,            # Batch size for training
        'optimizer': 'Adam',         # Optimizer type: 'SGD', 'Adam', 'AdamW'

        # === Optimizer Hyperparameters ===
        'learning_rate': 0.001,      # Initial learning rate
        'momentum': 0.9,             # Momentum for SGD (if used)
        'weight_decay': 0.0005,      # Weight decay for AdamW (if used)

        # === Augmentation and Logging ===
        'augment': True,             # Apply data augmentation
        'verbose': True,             # Enable verbose output

        # === Project Configuration ===
        'project_name': "volleyball_tracker_training",  # Weights & Biases project name
        'save_dir': "volleyball_tracker_training",      # Directory to save metrics and results
    }

    # ==============================
    #       Start Training
    # ==============================

    # Start the training and evaluation process with the defined configuration
    train_and_evaluate(config)
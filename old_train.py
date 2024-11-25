import argparse
from pathlib import Path
from ultralytics import YOLO
import wandb
import pandas as pd


def save_metrics_to_file(metrics, save_dir, experiment, avg_box_loss, avg_cls_loss, avg_dfl_loss):
    """
    Saves the evaluation metrics to a text file.

    Args:
        metrics (DetMetrics): Metrics object returned by model.val().
        save_dir (str or Path): Directory where the metrics file will be saved.
        experiment (str): Experiment name.
        avg_box_loss (float): Average box loss over all epochs.
        avg_cls_loss (float): Average classification loss over all epochs.
        avg_dfl_loss (float): Average DFL loss over all epochs.
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
        if avg_box_loss is not None:
            f.write(f"  - Average Box Loss: {avg_box_loss:.4f}\n")
        else:
            f.write("  - Average Box Loss: N/A\n")

        if avg_cls_loss is not None:
            f.write(f"  - Average Classification Loss: {avg_cls_loss:.4f}\n")
        else:
            f.write("  - Average Classification Loss: N/A\n")

        if avg_dfl_loss is not None:
            f.write(f"  - Average DFL Loss: {avg_dfl_loss:.4f}\n")
        else:
            f.write("  - Average DFL Loss: N/A\n")
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


def train_and_evaluate(data_yaml, model_name='pretrained_yolo/yolov8m.pt', epochs=20, imgsz=640,
                       project_name='volleyball_tracker_training'):
    """
    Trains YOLOv8 on a custom dataset and evaluates on the test set.
    """
    # Ensure wandb is initialized before training
    experiment = f"train_epochs{epochs}_model{model_name}"

    if wandb.run is None:
        wandb.init(project=project_name, name=experiment, resume='allow')

    # Load the model
    model = YOLO(model_name)

    # Train the model (wandb integration is automatic)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,  # experiment different values like 416
        val=True,
        project=project_name,
        name=experiment,
        exist_ok=True,
        verbose=True,
        augment=True,
        batch=16,
        optimizer='Adam'  # experiment with SGD, Adam or AdamW
    )

    # Read loss metrics from results.csv
    # Path to the results.csv file
    results_file = Path(project_name + '/' + str(experiment) + '/results.csv')

    # Initialize loss metrics
    avg_box_loss = avg_cls_loss = avg_dfl_loss = None

    # Check if the results.csv file exists
    if results_file.exists():
        # Read the results.csv file
        df = pd.read_csv(results_file)

        # Access loss metrics
        box_loss_list = df['train/box_loss'].tolist()
        cls_loss_list = df['train/cls_loss'].tolist()
        dfl_loss_list = df['train/dfl_loss'].tolist()

        # Calculate average loss metrics over all epochs
        avg_box_loss = sum(box_loss_list) / len(box_loss_list)
        avg_cls_loss = sum(cls_loss_list) / len(cls_loss_list)
        avg_dfl_loss = sum(dfl_loss_list) / len(dfl_loss_list)

        # Print the loss metrics
        print("Average Training Losses:")
        print(f"  - Box Loss: {avg_box_loss:.4f}")
        print(f"  - Classification Loss: {avg_cls_loss:.4f}")
        print(f"  - DFL Loss: {avg_dfl_loss:.4f}")
    else:
        print(f"Results file not found at {results_file}")

    # Evaluate the model on the test set
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=imgsz,
        save=True,
        save_txt=True,
        save_json=True,
        plots=True,
        verbose=True,
    )

    # Save metrics to a text file, including loss metrics
    save_metrics_to_file(metrics, metrics.save_dir, experiment, avg_box_loss, avg_cls_loss, avg_dfl_loss)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    epochs = 50

    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv8 on a custom dataset.")
    parser.add_argument('--data', type=str,
                        default="/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/data.yaml",
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pre-trained model to use')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, default='volleyball_tracker_training', help='Project name')

    args = parser.parse_args()

    train_and_evaluate(args.data, args.model, args.epochs, args.imgsz, args.project)

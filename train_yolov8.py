import argparse
from pathlib import Path
from ultralytics import YOLO
import wandb


def train_and_evaluate(data_yaml, model_name='yolov8n.pt', epochs=20, imgsz=640,
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
        imgsz=imgsz,
        val=True,
        project=project_name,
        name=experiment,
        exist_ok=True,
        verbose=True
    )

    # Evaluate the model on the test set
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=imgsz,
        save=True,
        save_txt=True,
        save_json=True,
        plots=True,
        verbose=True
    )

    # Save metrics to a text file
    save_metrics_to_file(metrics, metrics.save_dir, experiment)

    # Finish wandb run
    wandb.finish()


def save_metrics_to_file(metrics, save_dir, experiment):
    """
    Saves the evaluation metrics to a text file.

    Args:
        experiment (str): experiment name.
        metrics (DetMetrics): Metrics object returned by model.val().
        save_dir (str or Path): Directory where the metrics file will be saved.
    """
    name = f'metrics_report{experiment}.txt'
    metrics_file = Path(save_dir) / name
    with open(metrics_file, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")

        # Overall Metrics
        f.write(f"Metrics for the model:\n")
        f.write(f"  - Model name: {experiment}\n\n")
        f.write("Overall Performance:\n")

        # Access metrics via attributes, not methods
        f.write(f"  - Precision (P): {metrics.box.mp:.4f}\n")
        f.write(f"  - Recall (R): {metrics.box.mr:.4f}\n")
        f.write(f"  - Mean Average Precision @ IoU=0.5 (mAP@0.5): {metrics.box.map50:.4f}\n")
        f.write(f"  - Mean Average Precision @ IoU=0.5:0.95 (mAP@0.5:0.95): {metrics.box.map:.4f}\n\n")

        # Losses
        f.write("Losses:\n")
        if hasattr(metrics, 'box_loss'):
            f.write(f"  - Box Loss: {metrics.box_loss:.4f}\n")
        else:
            f.write("  - Box Loss: N/A\n")

        if hasattr(metrics, 'cls_loss'):
            f.write(f"  - Classification Loss: {metrics.cls_loss:.4f}\n")
        else:
            f.write("  - Classification Loss: N/A\n")

        if hasattr(metrics, 'dfl_loss'):
            f.write(f"  - DFL Loss: {metrics.dfl_loss:.4f}\n")
        else:
            f.write("  - DFL Loss: N/A\n")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv8 on a custom dataset.")
    parser.add_argument('--data', type=str,
                        default="/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/data.yaml",
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pre-trained model to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, default='volleyball_tracker_training', help='Project name')

    args = parser.parse_args()

    train_and_evaluate(args.data, args.model, args.epochs, args.imgsz, args.project)
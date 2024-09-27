import argparse
from ultralytics import YOLO

def train_and_evaluate(data_yaml, model_name='yolov8n.pt', epochs=50, imgsz=640, project_name='volleyball_tracker_training'):
    """
    Trains YOLOv8 on a custom dataset and evaluates on the test set.
    """
    # Load the model
    model = YOLO(model_name)

    # Train the model (wandb integration is automatic)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        val=True,
        project=project_name,
        name='train',
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

    # Print the evaluation metrics
    print("\nEvaluation Metrics on Test Set:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv8 on a custom dataset.")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pre-trained model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, default='volleyball_tracker_training', help='Project name')

    args = parser.parse_args()

    train_and_evaluate(args.data, args.model, args.epochs, args.imgsz, args.project)
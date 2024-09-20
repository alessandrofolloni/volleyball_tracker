import argparse
from ultralytics import YOLO


def train_and_evaluate(data_yaml, model_name='yolov8n.pt', epochs=50, imgsz=640, project_name='YOLOv8-Training'):
    """
    Trains YOLOv8 on a custom dataset and evaluates on the test set.

    Args:
        data_yaml (str): Path to the data YAML file.
        model_name (str): Pre-trained YOLOv8 model to start from.
        epochs (int): Number of training epochs.
        imgsz (int): Image size for training.
        project_name (str): Name of the wandb project.
    """
    # Import wandb and initialize
    import wandb
    wandb.init(project=project_name)

    # Load the model
    model = YOLO(model_name)

    # Train the model with wandb integration
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        val=True,  # Perform validation
        project='runs/detect',
        name='train',
        exist_ok=True,
        verbose=True,
        # Enable wandb logging
        callbacks=[wandb.WandbCallback()]
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

    # Log evaluation metrics to wandb
    wandb.log(metrics)

    # Finish wandb run
    wandb.finish()

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
    parser.add_argument('--project', type=str, default='YOLOv8-Training', help='wandb project name')

    args = parser.parse_args()

    train_and_evaluate(args.data, args.model, args.epochs, args.imgsz, args.project)

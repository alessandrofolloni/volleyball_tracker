import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import wandb


def train_and_evaluate(data_yaml, model_name='yolov8n.pt', epochs=50, imgsz=640,
                       project_name='volleyball_tracker_training'):
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
        name='train 3',
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

    # Log images comparing predictions and ground truth
    log_comparison_images(model, data_yaml, metrics.save_dir)


def log_comparison_images(model, data_yaml, save_dir):
    # Load class names from data.yaml
    import yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data.get('names', [])

    # Directory paths
    images_dir = data['test']
    labels_dir = images_dir.replace('images', 'labels')
    predictions_dir = save_dir / 'labels'

    # Get list of images
    image_paths = [p for p in Path(images_dir).glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    comparison_images = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape

        # Draw ground truth boxes
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    draw_box(img, x_center, y_center, width, height, w, h, color=(0, 255, 0),
                             label=class_names[int(class_id)])

        # Draw predicted boxes
        pred_path = Path(predictions_dir) / (img_path.stem + '.txt')
        if pred_path.exists():
            with open(pred_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height, conf = map(float, line.strip().split())
                    draw_box(img, x_center, y_center, width, height, w, h, color=(0, 0, 255),
                             label=class_names[int(class_id)], conf=conf)

        # Convert BGR to RGB for Wandb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Log the image
        comparison_images.append(wandb.Image(img_rgb, caption=img_path.name))

    # Log to Wandb
    wandb.log({"Predictions vs Ground Truth": comparison_images})


def draw_box(img, x_center, y_center, width, height, img_w, img_h, color, label='', conf=None):
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label_text = label
    if conf is not None:
        label_text += f' {conf:.2f}'
    if label_text:
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv8 on a custom dataset.")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pre-trained model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--project', type=str, default='volleyball_tracker_training', help='Project name')

    args = parser.parse_args()

    train_and_evaluate(args.data, args.model, args.epochs, args.imgsz, args.project)

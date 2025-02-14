import argparse
import os
import random
from ultralytics import YOLO


def track_players(video_path, model_path='volleyball_tracker_training/train_epochs50_modelyolov8m_bs16_optAdam/weights/best.pt',
                  output_dir='runs/track/', imgsz=640,
                  tracker_cfg='/Users/alessandrofolloni/PycharmProjects'
                              '/volleyball_tracker/.venv/lib/python3.10/site-packages'
                              '/ultralytics/cfg/trackers/bytetrack.yaml'):
    """
    Performs player tracking on a video using a fine-tuned YOLOv8 model.

    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to the trained YOLOv8 model.
        output_dir (str): Directory to save the output.
        imgsz (int): Image size for inference.
        tracker_cfg (str): Tracker configuration file.
    """
    # Load the trained model
    model = YOLO(model_path)

    # Run tracking on the video
    model.track(
        source=video_path,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name='results',
        exist_ok=True,
        imgsz=imgsz,
        tracker=tracker_cfg
    )


if __name__ == "__main__":
    file = random.choice(os.listdir("/Users/alessandrofolloni/PycharmProjects"
                                    "/volleyball_tracker/videos/Converted_short"))
    file_path = 'videos/Converted_short/' + file

    parser = argparse.ArgumentParser(description="Track players in a video using YOLOv8.")
    parser.add_argument('--video', type=str, default=file_path, help='Path to input video')
    parser.add_argument('--model', type=str, default='volleyball_tracker_training/train_epochs50_modelyolov8m_bs16_optAdam/weights/best.pt', help='Path to trained model')
    parser.add_argument('--output', type=str, default='runs/track/', help='Output directory')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker configuration')

    args = parser.parse_args()

    track_players(args.video, args.model, args.output, args.imgsz, args.tracker)

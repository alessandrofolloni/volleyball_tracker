import cv2
import os
import glob
import random


def extract_random_frames(video_files, total_frames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_frames = []

    # Estrai tutti i frame dai video
    for idx, video_file in enumerate(video_files):
        cap = cv2.VideoCapture(video_file)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append((frame, f"video{idx + 1}_frame{frame_index}"))
            frame_index += 1
        cap.release()

    selected_frames = random.sample(all_frames, min(total_frames, len(all_frames)))

    # Salva i frame selezionati
    for count, (frame, frame_id) in enumerate(selected_frames):
        frame_path = os.path.join(output_dir, f"{frame_id}_{count}.jpg")
        cv2.imwrite(frame_path, frame)

    cv2.destroyAllWindows()
    print(f"Frame extraction completed. Total frames saved: {len(selected_frames)}")


# Directories
video_dir = '/Users/alessandrofolloni/Desktop/Unibo/PW ML4CV/videos/Converted'
output_dir = '/Users/alessandrofolloni/Desktop/Unibo/PW ML4CV/videos/frames_extracted'

# Find videos inside video_dir
video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
total_frames = 1000

extract_random_frames(video_files, total_frames, output_dir)
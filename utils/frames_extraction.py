import cv2
import os
import glob
import random


class VideoFrameExtractor:
    def __init__(self, video_dir, output_dir, total_frames):
        """
        Initialize the VideoFrameExtractor class with directories and parameters.

        :param video_dir: Directory containing video files.
        :param output_dir: Directory where extracted frames will be saved.
        :param total_frames: Total number of random frames to extract.
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.total_frames = total_frames

        # Find videos in the specified directory
        self.video_files = glob.glob(os.path.join(self.video_dir, '*.mp4'))

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_random_frames(self):
        """
        Extract random frames from videos in the specified directory.
        """
        all_frames = []

        # Extract all frames from the videos
        for idx, video_file in enumerate(self.video_files):
            cap = cv2.VideoCapture(video_file)
            frame_index = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append((frame, f"video{idx + 1}_frame{frame_index}"))
                frame_index += 1
            cap.release()

        # Select random frames
        selected_frames = random.sample(all_frames, min(self.total_frames, len(all_frames)))

        # Save the selected frames
        self._save_frames(selected_frames)

        cv2.destroyAllWindows()
        print(f"Frame extraction completed. Total frames saved: {len(selected_frames)}")

    def _save_frames(self, selected_frames):
        """
        Save selected frames to the output directory.

        :param selected_frames: List of tuples containing frames and their corresponding IDs.
        """
        for count, (frame, frame_id) in enumerate(selected_frames):
            frame_path = os.path.join(self.output_dir, f"{frame_id}_{count}.jpg")
            cv2.imwrite(frame_path, frame)

    def get_video_files(self):
        """
        Get the list of video files in the video directory.

        :return: List of video file paths.
        """
        return self.video_files

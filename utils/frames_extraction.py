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
        # Get total number of frames for each video
        total_frames_per_video = []
        for video_file in self.video_files:
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_per_video.append(total_frames)
            cap.release()

        # Compute cumulative frames
        cumulative_frames = []
        total_frames_cumulative = 0
        for frames in total_frames_per_video:
            total_frames_cumulative += frames
            cumulative_frames.append(total_frames_cumulative)

        total_frames_all_videos = total_frames_cumulative

        # Generate random frame indices
        total_frames_to_extract = min(self.total_frames, total_frames_all_videos)
        random_frame_indices = sorted(random.sample(range(total_frames_all_videos), total_frames_to_extract))

        # Map frame indices to videos
        frames_per_video_to_extract = {}
        video_start_frame = 0
        for idx, video_file in enumerate(self.video_files):
            video_frames = total_frames_per_video[idx]
            video_end_frame = video_start_frame + video_frames

            # Get frame indices for this video
            frames_in_this_video = [f - video_start_frame for f in random_frame_indices
                                    if video_start_frame <= f < video_end_frame]

            if frames_in_this_video:
                frames_per_video_to_extract[video_file] = frames_in_this_video

            video_start_frame = video_end_frame

        # Now extract frames
        frame_count = 0
        for video_file, frame_indices in frames_per_video_to_extract.items():
            cap = cv2.VideoCapture(video_file)
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_id = f"{os.path.basename(video_file)}_frame{frame_idx}"
                    frame_path = os.path.join(self.output_dir, f"{frame_id}_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                else:
                    print(f"Failed to read frame {frame_idx} from {video_file}")
            cap.release()

        cv2.destroyAllWindows()
        print(f"Frame extraction completed. Total frames saved: {frame_count}")

    def get_video_files(self):
        """
        Get the list of video files in the video directory.

        :return: List of video file paths.
        """
        return self.video_files

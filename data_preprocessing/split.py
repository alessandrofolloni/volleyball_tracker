import os
from moviepy.video.io.VideoFileClip import VideoFileClip


def split_video(video_path, output_folder, max_duration=600):
    """
    Splits the video at video_path into multiple segments of max_duration seconds.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory where the output video segments will be saved.
        max_duration (int): Maximum duration (in seconds) of each segment.
    """
    # Get the video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Load the video
    with VideoFileClip(video_path) as video:
        duration = int(video.duration)
        # Calculate number of splits needed
        num_splits = (duration + max_duration - 1) // max_duration
        for i in range(num_splits):
            start_time = i * max_duration
            end_time = min((i + 1) * max_duration, duration)
            # Extract the subclip
            subclip = video.subclip(start_time, end_time)
            # Create output filename
            output_filename = f"{video_name}_part{i + 1}.mp4"
            output_path = os.path.join(output_folder, output_filename)
            # Write the subclip to file
            subclip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=f"{output_filename}_temp_audio.m4a",
                remove_temp=True,
                logger=None
            )


def split_videos_in_folder(input_folder, output_folder, max_duration=600):
    """
    Splits all video files in input_folder into multiple segments of max_duration seconds.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Directory where the output video segments will be saved.
        max_duration (int): Maximum duration (in seconds) of each segment.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported video file extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(input_folder, filename)
            print(f"Processing {video_path}...")
            split_video(video_path, output_folder, max_duration)
            print(f"Finished processing {video_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split videos in a folder into segments of maximum 10 minutes.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder containing input videos.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the folder where output videos will be saved.')
    parser.add_argument('--max_duration', type=int, default=300,
                        help='Maximum duration (in seconds) of each video segment.')

    args = parser.parse_args()

    split_videos_in_folder(args.input_folder, args.output_folder, args.max_duration)


'''
Usage
python utils/split.py --input_folder /Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/Converted 
            --output_folder /Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/Converted_short
'''
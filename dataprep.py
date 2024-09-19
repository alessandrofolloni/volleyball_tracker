from utils.frames_extraction import VideoFrameExtractor
from utils.frame_review import ImageReviewer

video_dir = '/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/Converted'
output_dir = '/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/frames_extracted'
total_frames = 1000

# Extract random frames
extractor = VideoFrameExtractor(video_dir, output_dir, total_frames)
extractor.extract_random_frames()

# Review and delete images
reviewer = ImageReviewer(output_dir)
reviewer.review_and_delete_images()
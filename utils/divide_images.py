import os
import shutil


def divide_images(input_dir, output_base_dir):
    """
    Divides images from the input directory into three folders evenly.

    :param input_dir: Directory containing the images to divide.
    :param output_base_dir: Base directory where the three output folders will be created.
    """
    # Get list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    total_images = len(image_files)

    if total_images == 0:
        print("No images found in the input directory.")
        return

    images_per_folder = total_images // 3
    extra_images = total_images % 3  # This will be 0, 1, or 2

    # Create output directories
    output_dirs = []
    for i in range(1, 4):
        output_dir = os.path.join(output_base_dir, f'images_folder_{i}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dirs.append(output_dir)

    # Sort image files to ensure consistent distribution
    image_files.sort()

    # Assign images to folders
    start_idx = 0
    for i in range(3):
        num_images_in_folder = images_per_folder + (1 if i < extra_images else 0)
        end_idx = start_idx + num_images_in_folder
        folder_images = image_files[start_idx:end_idx]
        for image_file in folder_images:
            src_path = os.path.join(input_dir, image_file)
            dest_path = os.path.join(output_dirs[i], image_file)
            shutil.copyfile(src_path, dest_path)
        start_idx = end_idx
        print(f"Copied {len(folder_images)} images to {output_dirs[i]}")

    print("Images have been divided evenly into 3 folders.")


if __name__ == "__main__":
    input_dir = '/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/frames_extracted'
    output_base_dir = '/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos'
    divide_images(input_dir, output_base_dir)

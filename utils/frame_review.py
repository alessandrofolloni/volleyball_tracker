import cv2
import os
import glob


def review_and_delete_images(image_dir):
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Could not read {image_file}. Skipping.")
            continue

        cv2.imshow('Image', image)
        cv2.waitKey(1)

        while True:
            user_input = input(f"Do you want to delete {os.path.basename(image_file)}? (y/n/q): ").lower()
            if user_input in ['y', 'n', 'q']:
                break
            print("Invalid input. Please enter 'y', 'n', or 'q'.")

        if user_input == 'y':
            os.remove(image_file)
            print(f"{os.path.basename(image_file)} deleted.")
        elif user_input == 'n':
            print(f"{os.path.basename(image_file)} kept.")
        elif user_input == 'q':
            print("Exiting...")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print("Review completed.")


# Directories
image_dir = '/Users/alessandrofolloni/Desktop/Unibo/PW ML4CV/videos/frames_extracted'

review_and_delete_images(image_dir)
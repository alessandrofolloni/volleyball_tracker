import cv2
import os
import glob


class ImageReviewer:
    def __init__(self, image_dir):
        """
        Initialize the ImageReviewer class with the directory containing images.

        :param image_dir: Directory containing image files to review.
        """
        self.image_dir = image_dir
        self.image_files = glob.glob(os.path.join(self.image_dir, '*.jpg'))

    def review_and_delete_images(self):
        """
        Review each image in the directory and prompt the user to delete or keep it.
        """
        for image_file in self.image_files:
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
                self._delete_image(image_file)
            elif user_input == 'n':
                print(f"{os.path.basename(image_file)} kept.")
            elif user_input == 'q':
                print("Exiting...")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()
        print("Review completed.")

    def _delete_image(self, image_file):
        """
        Delete the specified image file.

        :param image_file: The path of the image file to delete.
        """
        os.remove(image_file)
        print(f"{os.path.basename(image_file)} deleted.")


# Usage
image_dir = '/Users/alessandrofolloni/Desktop/Unibo/PW ML4CV/videos/frames_extracted'

# Create an instance of ImageReviewer
reviewer = ImageReviewer(image_dir)

# Review and delete images
reviewer.review_and_delete_images()
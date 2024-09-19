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
        Press Enter to keep the image, any other input to delete.
        """
        for image_file in self.image_files:
            image = cv2.imread(image_file)
            if image is None:
                print(f"Could not read {image_file}. Skipping.")
                continue

            cv2.imshow('Image', image)
            cv2.waitKey(1)

            user_input = input(f"Press Enter to keep '{os.path.basename(image_file)}', or any other key to delete: ")

            cv2.destroyAllWindows()

            if user_input == '':
                print(f"'{os.path.basename(image_file)}' kept.")
            else:
                self._delete_image(image_file)

        print("Review completed.")

    def _delete_image(self, image_file):
        """
        Delete the specified image file.

        :param image_file: The path of the image file to delete.
        """
        os.remove(image_file)
        print(f"'{os.path.basename(image_file)}' deleted.")
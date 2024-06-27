import numpy as np
import cv2

class ImageStitcher:
    def __init__(self, image1, image2, homography_matrix):
        """
        Initializes the ImageStitcher with two images and a homography matrix.
        
        Parameters:
        image1 (numpy array): The first image to be stitched.
        image2 (numpy array): The second image to be stitched.
        homography_matrix (numpy array): The homography matrix that transforms image2 into the perspective of image1.
        """
        self.image1 = image1
        self.image2 = image2
        self.homography_matrix = homography_matrix

    def find_intersection(self):
        """
        Finds the intersection area of the two images using the homography matrix.
        
        Returns:
        intersection (tuple): The coordinates of the intersection area.
        """

        pass

    def split_into_patches(self, intersection, patch_size):
        """
        Splits the intersection area into smaller patches that can fit into GPU memory.
        
        Parameters:
        intersection (tuple): The coordinates of the intersection area.
        patch_size (tuple): The size of each patch.
        
        Returns:
        patches (list): A list of smaller patches from the intersection area.
        """
        patches = []
        

        return patches

    def process_patches(self, patches):
        """
        Processes each patch using RAFT for detailed stitching.
        
        Parameters:
        patches (list): A list of smaller patches from the intersection area.
        
        Returns:
        processed_patches (list): A list of processed patches.
        """
        processed_patches = []
        for patch in patches:
            # todo: process each patch using RAFT
            pass
        return processed_patches

    def stitch_images(self, processed_patches):
        """
        Stitches the processed patches together to form the final stitched image.
        
        Parameters:
        processed_patches (list): A list of processed patches.
        
        Returns:
        final_image (numpy array): The final stitched image.
        """
        final_image = None
        
        return final_image

    def stitch(self, patch_size):
        """
        Main method to perform the stitching of the two images.
        
        Parameters:
        patch_size (tuple): The size of each patch.
        
        Returns:
        final_image (numpy array): The final stitched image.
        """
        intersection = self.find_intersection()
        patches = self.split_into_patches(intersection, patch_size)
        processed_patches = self.process_patches(patches)
        final_image = self.stitch_images(processed_patches)
        return final_image
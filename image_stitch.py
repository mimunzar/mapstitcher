import numpy as np
import cv2
import argparse

class ImageStitcher:
    debug = True

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

        corners1 = np.array([[0, 0, 1], [self.image1.shape[1], 0, 1], [self.image1.shape[1], self.image1.shape[0], 1], [0, self.image1.shape[0], 1]]) 
        # corner coordinates of image1
        corners2 = np.array([[0, 0, 1], [self.image2.shape[1], 0, 1], [self.image2.shape[1], self.image2.shape[0], 1], [0, self.image2.shape[0], 1]]) 
        # corner coordinates of image2

        corners2 = corners2 @ self.homography_matrix.T
        corners2 = corners2 / corners2[:, 2].reshape(-1, 1)
        # transform corners of image2 to image1's perspective
        corners1 = corners1[:, :2] / corners1[:, 2, np.newaxis]
        corners2 = corners2[:, :2] / corners2[:, 2, np.newaxis]
        # homogenous -> cartesian

        contour1 = np.array(corners1, dtype=np.float32).reshape((-1, 1, 2))
        contour2 = np.array(corners2, dtype=np.float32).reshape((-1, 1, 2))
        # Convert corners to contours - cv2

        intersection = cv2.intersectConvexConvex(contour1, contour2)
        if intersection[1] is None:
            self.intersection = None
        # Find the intersection of the two contours

        intersection_points = intersection[1]
        # Get the intersecting points

        x_min = np.min(intersection_points[:, 0, 0])
        y_min = np.min(intersection_points[:, 0, 1])
        x_max = np.max(intersection_points[:, 0, 0])
        y_max = np.max(intersection_points[:, 0, 1])
        # Find the bounding box of the intersection points

        self.intersection = (x_min, y_min, x_max, y_max)
        if ImageStitcher.debug:
            print (self.intersection)

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
    
def parse_args():
    """
    Dummy function to parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Stitcher")
    parser.add_argument('--img0', required=True, help="Path to the first image")
    parser.add_argument('--img1', required=True, help="Path to the second image")
    parser.add_argument('--homography', required=True, help="Path to the homography matrix in .npy format")
    return parser.parse_args()

def main():
    """
    Dummy main function to test the ImageStitcher class.
    """
    args = parse_args()
    
    # Load images
    img0 = cv2.imread(args.img0)
    img1 = cv2.imread(args.img1)
    
    if img0 is None or img1 is None:
        print("Error: One or both images could not be loaded.")
        sys.exit(1)
    
    # Load homography matrix
    homography_matrix = np.load(args.homography)
    
    # Initialize the ImageStitcher
    stitcher = ImageStitcher(img0, img1, homography_matrix)
    
    # Perform the stitching
    patch_size = (512, 512)  # Example patch size, you can adjust this
    final_image = stitcher.stitch(patch_size)
    
    # Save or display the final stitched image
    if final_image is not None:
        cv2.imwrite('stitched_image.jpg', final_image)
        print("Stitched image saved as 'stitched_image.jpg'.")
    else:
        print("Error: Stitching failed.")

if __name__ == '__main__':
    main()
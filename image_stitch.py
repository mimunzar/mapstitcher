import numpy as np
import cv2
import argparse

class ImageStitcher:
    debug = True

    def __init__(self, image1, image2, H):
        """
        Initializes the ImageStitcher with two images and a homography matrix.
        
        Parameters:
        image1 (numpy array): The first image to be stitched.
        image2 (numpy array): The second image to be stitched.
        H (numpy array): The homography matrix that transforms image2 into the perspective of image1.
        """
        self.image1 = image1
        self.image2 = image2
        self.H = H

    def find_intersection(self):
        """
        Finds the intersection area of the two images using the homography matrix.
        
        Computes:
        intersection (tuple): The coordinates of the intersection area.
        intersection_center (numpy array): The center of the intersection area.
        intersection_normal (numpy array): The normal vector of the intersection area.
        """

        corners1 = np.array([[0, 0, 1], [self.image1.shape[1], 0, 1], [self.image1.shape[1], self.image1.shape[0], 1], [0, self.image1.shape[0], 1]]) 
        # corner coordinates of image1
        corners2 = np.array([[0, 0, 1], [self.image2.shape[1], 0, 1], [self.image2.shape[1], self.image2.shape[0], 1], [0, self.image2.shape[0], 1]]) 
        # corner coordinates of image2

        corners2 = corners2 @ self.H.T
        corners2 = corners2 / corners2[:, 2].reshape(-1, 1)
        # transform corners of image2 to image1 perspective
        corners1 = corners1[:, :2] / corners1[:, 2, np.newaxis]
        corners2 = corners2[:, :2] / corners2[:, 2, np.newaxis]
        # homogenous -> cartesian

        contour1 = np.array(corners1, dtype=np.float32).reshape((-1, 1, 2))
        contour2 = np.array(corners2, dtype=np.float32).reshape((-1, 1, 2))
        # convert corners to contours - cv2

        intersection = cv2.intersectConvexConvex(contour1, contour2)
        if intersection[1] is None:
            intersection = None
            intersection_center = None
            intersection_normal = None
            return intersection, intersection_center, intersection_normal
        # find the intersection of the two contours

        intersection_points = intersection[1]
        # get the intersecting points
        intersection_center = np.mean(intersection_points, axis=0)[0]
        # get intersection center
        intersection_normal = np.mean(corners2, axis=0) - np.mean(corners1, axis=0)
        magnitude = np.linalg.norm(intersection_normal)
        if magnitude != 0:
            intersection_normal /= magnitude
        # get intersection normal

        x_min = np.min(intersection_points[:, 0, 0])
        y_min = np.min(intersection_points[:, 0, 1])
        x_max = np.max(intersection_points[:, 0, 0])
        y_max = np.max(intersection_points[:, 0, 1])
        # find the bounding box of the intersection points

        intersection = (x_min, y_min, x_max, y_max)
        if ImageStitcher.debug:
            print (intersection)
            print (intersection_center)
            print (intersection_normal)

        return intersection, intersection_center, intersection_normal

    def split_into_patches(self, intersection, intersection_center, intersection_normal, patch_size):
        """
        Splits the intersection area into smaller patches that can fit into GPU memory.
        
        Parameters:
        intersection (tuple): The coordinates of the intersection area.
        intersection_center (numpy array): The center of the intersection area.
        intersection_normal (numpy array): The normal vector of the intersection area.
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
        intersection, intersection_center, intersection_normal = self.find_intersection()
        patches = self.split_into_patches(intersection, intersection_center, intersection_normal, patch_size)
        processed_patches = self.process_patches(patches)
        final_image = self.stitch_images(processed_patches)
        return final_image
    
def parse_args():
    """
    Dummy function to parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Stitcher")
    parser.add_argument('--img1', required=True, help="Path to the first image")
    parser.add_argument('--img2', required=True, help="Path to the second image")
    parser.add_argument('--H', required=True, help="Path to the homography matrix in .npy format")
    return parser.parse_args()

def main():
    """
    Dummy main function to test the ImageStitcher class.
    """
    args = parse_args()
    
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    # load images
    
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        sys.exit(1)
    
    H = np.load(args.H)
    # load homography matrix
    
    stitcher = ImageStitcher(img1, img2, H)
    # initialize the ImageStitcher

    patch_size = (512, 512)  # Example patch size
    final_image = stitcher.stitch(patch_size)
    # perform the stitching
    
    if final_image is not None:
        cv2.imwrite('stitched_image.jpg', final_image)
        print("Stitched image saved as 'stitched_image.jpg'.")
    else:
        print("Error: Stitching failed.")
    # save or display the final stitched image

if __name__ == '__main__':
    main()
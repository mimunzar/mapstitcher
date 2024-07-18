import numpy as np
import cv2
import argparse
import math

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt

from optical_flow_raft import OpticalFlow_RAFT
from optical_flow_cv import OpticalFlow_CV

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
        self.x_dominant = True

        #self.flow_estimator = OpticalFlow_RAFT()
        self.flow_estimator = OpticalFlow_CV()

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

        self.final_image = np.zeros((int(max(np.max(corners1[:, 1]), np.max(corners2[:, 1]))), int(max(np.max(corners1[:, 0]), np.max(corners2[:, 0]))), 3), dtype=np.uint8)
        # todo: take into account negative X and Y after homography
        # create final image shape

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
        
        x_min, y_min, x_max, y_max = intersection
        x0, y0 = intersection_center
        dx, dy = [-intersection_normal[1], intersection_normal[0]]
        x_dominant = True

        intersections = []
        if dx != 0:
            t = (x_min - x0) / dx
            y = y0 + t * dy
            if y_min <= y <= y_max:
                intersections.append((x_min, y))
        
            t = (x_max - x0) / dx
            y = y0 + t * dy
            if y_min <= y <= y_max:
                intersections.append((x_max, y))
        
        if(len(intersections) == 0):
            if dy != 0:
                t = (y_min - y0) / dy
                x = x0 + t * dx
                if x_min <= x <= x_max:
                    intersections.append((x, y_min))
            
            if dy != 0:
                t = (y_max - y0) / dy
                x = x0 + t * dx
                if x_min <= x <= x_max:
                    intersections.append((x, y_max))
            x_dominant = False

        self.x_dominant = x_dominant
        patch_width, patch_height = patch_size
        # get patch width and height

        print(intersection)

        if(x_dominant):
            for i in range(math.ceil((x_max - x_min) / patch_width)):
                x = x_min + i * patch_width
                y = y0 + (x - x0) * dy / dx
                #patch = (x, y - patch_height / 2, min(x + patch_width, x_max), y + patch_height / 2)
                patch = (x, y - patch_height / 2, x + patch_width, x_max, y + patch_height / 2)
                patches.append(patch)
        else:
            for i in range(math.ceil((y_max - y_min) / patch_height)):
                y = y_min + i * patch_height
                x = x0 + (y - y0) * dx / dy
                #patch = (x - patch_width / 2, y, x + patch_width / 2, min(y + patch_height, y_max))
                patch = (x - patch_width / 2, y, x + patch_width / 2, y + patch_height)
                patches.append(patch)
        # find intersection of intersection area and line given by centroid and normal

        print(patches)

        return patches

    def get_padded_patch(self, image, patch):
        x_min, y_min, x_max, y_max = patch

        # Calculate the padding amounts
        pad_left = max(0, -x_min)
        pad_top = max(0, -y_min)
        pad_right = max(0, x_max - image.shape[1])
        pad_bottom = max(0, y_max - image.shape[0])

        # Pad the image
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

        # Calculate new patch coordinates in the padded image
        x_min += pad_left
        y_min += pad_top
        x_max += pad_left
        y_max += pad_top

        # Extract the patch
        patch_img = padded_image[y_min:y_max, x_min:x_max]
        return patch_img

    def process_patches(self, patches, intersection_normal):
        """
        Processes each patch using RAFT for detailed stitching.
        
        Parameters:
        patches (list): A list of smaller patches from the intersection area.
        
        Returns:
        processed_patches (list): A list of processed patches.
        """
        processed_patches = []

        patch1_list = []
        patch2_list = []
        for patch in patches:
            patch1 = [int(patch[0]), int(patch[1]), int(patch[2]), int(patch[3])]
            # get patch1 coordinates
            patch2 = cv2.perspectiveTransform(np.array([[[patch[0], patch[1]], [patch[2], patch[1]], [patch[2], patch[3]], [patch[0], patch[3]]]], np.float32), np.linalg.inv(self.H))
            patch2 = np.squeeze(patch2)
            patch2 = [int(min(patch2[0][0], patch2[3][0])), int(min(patch2[0][1], patch2[1][1])), int(min(patch2[0][0], patch2[3][0]))+256, int(min(patch2[0][1], patch2[1][1]))+256]
            # get patch2 coordinates

            patch1_img = self.get_padded_patch(self.image1, patch1)
            patch2_img = self.get_padded_patch(self.image2, patch2)
            print(patch1_img.shape, patch2_img.shape)
            # extract image patches

            patch1_list.append(patch1_img)
            patch2_list.append(patch2_img)

        predicted_flows = self.flow_estimator.process_images(patch1_list, patch2_list)
        # get optical flow between patches

        for i, (patch1, patch2) in enumerate(zip(patch1_list, patch2_list)):
            flow_np = predicted_flows[i].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

            weight_array = np.zeros((patch1.shape[0], patch1.shape[1]), dtype=np.float32)
            for y in range(weight_array.shape[0]):
                for x in range(weight_array.shape[1]):
                    weight_array[y, x] = intersection_normal[0] * (x - weight_array.shape[1] / 2 + 0.5) + intersection_normal[1] * (y - weight_array.shape[0] / 2 + 0.5)

            # create weight array
            max_value = np.max(weight_array)
            min_value = np.min(weight_array)
            weight_array = (weight_array - min_value) / (max_value - min_value)
            # normalize weight array

            flow_x = flow_np[..., 0]
            flow_y = flow_np[..., 1]
            # get flow x and y
            for y in range(flow_x.shape[0]):
                for x in range(flow_x.shape[1]):
                    flow_x[y, x] = flow_x[y, x] * (1 - weight_array[y, x])
                    flow_y[y, x] = flow_y[y, x] * (1 - weight_array[y, x])
            # create flow gradient, so warped image ties to source image
            x, y = np.meshgrid(np.arange(flow_np.shape[1]), np.arange(flow_np.shape[0]))
            new_x = x + flow_x
            new_y = y + flow_y
            #patch2_wrap = cv2.remap(patch2, new_x.astype(np.float32), new_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
            patch2_wrap = cv2.remap(patch2, new_x.astype(np.float32), new_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            # create warped image
            valid_mask = (new_x >= 0) & (new_x < patch2.shape[1]) & (new_y >= 0) & (new_y < patch2.shape[0])
            # create mask

            blended = [patch2_wrap, valid_mask]
            #blended = np.array(patch1)
            #for y in range(weight_array.shape[0]):
            #    for x in range(weight_array.shape[1]):
            #        blended[y, x] = patch1[y, x] * (1 - weight_array[y, x]) + patch2_wrap[y, x] * weight_array[y, x]
            # create blended image

            processed_patches.append(blended)

            #cv2.imshow('patch1', patch1)
            #cv2.imshow('patch2', patch2)
            #cv2.imshow('patch2_wrap', patch2_wrap)
            #cv2.imshow('blended', blended)
            #cv2.waitKey(0)

        return processed_patches

    def stitch_images(self, patches, processed_patches, intersection_center, intersection_normal):
        """
        Stitches the processed patches together to form the final stitched image.
        
        Parameters:
        patches (list): A list of patch positions
        processed_patches (list): A list of processed patches.
        
        Returns:
        final_image (numpy array): The final stitched image.
        """
        final_image = None

        # warp image 2 with homography
        image1_warp = cv2.warpPerspective(self.image1, np.identity(3), (self.final_image.shape[1], self.final_image.shape[0]))
        image2_warp = cv2.warpPerspective(self.image2, self.H, (self.final_image.shape[1], self.final_image.shape[0]))
        # copy to self.final_image
        self.final_image = self.create_composite_image(image1_warp, image2_warp, intersection_center, intersection_normal)
        #cv2.imshow('w0', image1_warp)
        #cv2.imshow('w1', image2_warp)
        cv2.imshow('final', self.final_image)
        cv2.waitKey(0)
        for patch, [processed_patch, mask] in zip(patches, processed_patches):
            print(patch)

            x_min, y_min, x_max, y_max = patch
            h, w = processed_patch.shape[:2]
            x_min = int(x_min)
            y_min = int(y_min)
            # get patch coordinates

            roi_y1 = max(0, y_min)
            roi_y2 = min(self.final_image.shape[0], y_min + h)
            roi_x1 = max(0, x_min)
            roi_x2 = min(self.final_image.shape[1], x_min + w)
            # get roi coordinates
            
            patch_y1 = max(0, -y_min)
            patch_y2 = patch_y1 + (roi_y2 - roi_y1)
            patch_x1 = max(0, -x_min)
            patch_x2 = patch_x1 + (roi_x2 - roi_x1)
            # get patch coordinates -> this whole ordeal is because of out-of-bounds copying

            patch_transformed = processed_patch[patch_y1:patch_y2, patch_x1:patch_x2]
            valid_mask = mask[patch_y1:patch_y2, patch_x1:patch_x2]
            # get the transformed patch and mask

            roi = self.final_image[roi_y1:roi_y2, roi_x1:roi_x2]
            # get the roi from the final image

            valid_mask_expanded = np.repeat(valid_mask[:, :, np.newaxis], 3, axis=2)
            roi[valid_mask_expanded] = patch_transformed[valid_mask_expanded]
            # copy the transformed patch to the roi
            
            self.final_image[roi_y1:roi_y2, roi_x1:roi_x2] = roi
            # copy processed patches to final image

            cv2.imshow('final', self.final_image)
            cv2.waitKey(0)
        
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
        processed_patches = self.process_patches(patches, intersection_normal)
        final_image = self.stitch_images(patches, processed_patches, intersection_center, intersection_normal)
        return final_image
    
    def create_composite_image(self, img1, img2, point, normal):
        """
        Create a composite image with img1 on one side of the line and img2 on the other side.

        Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        point (tuple): A point on the line (x, y).
        normal (tuple): The normal to the line (nx, ny).

        Returns:
        np.ndarray: Composite image.
        """
        
        assert img1.shape == img2.shape, "Images must have the same size"
        # assert
        
        height, width = img1.shape[:2]
        # get dimensions

        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        # create meshgrid

        nx, ny = normal
        px, py = point
        # define line parameters

        mask = (nx * (x - px) + ny * (y - py)) < 0
        # create mask

        composite_image = np.where(mask[..., None], img1, img2)
        # Create the composite image

        return composite_image
    
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

    patch_size = (256, 256)  # Example patch size
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
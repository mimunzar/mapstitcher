import numpy as np
import cv2
import argparse
import math

import torch
import torch.nn.functional as F2

from libs.optical_flow_raft import OpticalFlow_RAFT

from libs.util import *
from libs.homography import *

import matplotlib.pyplot as plt

def create_composite_image(img1, img2, point, normal):
    """
    Create a composite image with img1 on one side of the line and img2 on the other side.

    Args:
    img1 (torch.Tensor): First image tensor of shape [C, H, W].
    img2 (torch.Tensor): Second image tensor of shape [C, H, W].
    point (tuple): A point on the line (x, y).
    normal (tuple): The normal to the line (nx, ny).

    Returns:
    torch.Tensor: Composite image tensor of shape [C, H, W].
    """
    from torch import meshgrid
    from torch import arange
    from torch import where
    assert img1.shape == img2.shape, "Images must have the same size"
    
    height, width = img1.shape[1:]
    y, x = meshgrid(arange(height, device=img1.device), arange(width, device=img1.device), indexing='ij')
    
    nx, ny = normal
    px, py = point
    
    mask = (nx * (x - px) + ny * (y - py)) < 0
    composite_image = where(mask.unsqueeze(0), img1, img2)
    
    return composite_image

class ImageStitcher:
    def __init__(self, image1, image2, H, debug=False):
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
        self.debug = debug

        self.flow_estimator = OpticalFlow_RAFT(self.debug)
        #self.flow_estimator = OpticalFlow_CV()

    def find_intersection(self):
        """
        Finds the intersection area of the two images using the homography matrix.
        
        Computes:
        intersection (tuple): The coordinates of the intersection area.
        intersection_center (numpy array): The center of the intersection area.
        intersection_normal (numpy array): The normal vector of the intersection area.
        """
        corners1 = np.array([[0, 0, 1], [self.image1.shape[2], 0, 1], [self.image1.shape[2], self.image1.shape[1], 1], [0, self.image1.shape[1], 1]]) 
        # corner coordinates of image1
        corners2 = np.array([[0, 0, 1], [self.image2.shape[2], 0, 1], [self.image2.shape[2], self.image2.shape[1], 1], [0, self.image2.shape[1], 1]]) 
        # corner coordinates of image2

        corners2 = corners2 @ self.H.T
        corners2 = corners2 / corners2[:, 2].reshape(-1, 1)
        # transform corners of image2 to image1 perspective
        corners1 = corners1[:, :2] / corners1[:, 2, np.newaxis]
        corners2 = corners2[:, :2] / corners2[:, 2, np.newaxis]
        # homogenous -> cartesian

        self.final_image_dimensions = (int(max(np.max(corners1[:, 1]), np.max(corners2[:, 1]))), int(max(np.max(corners1[:, 0]), np.max(corners2[:, 0]))))
        # store final image dimensions according to the corners
        self.final_image = np.zeros((int(max(np.max(corners1[:, 1]), np.max(corners2[:, 1]))), int(max(np.max(corners1[:, 0]), np.max(corners2[:, 0]))), 3), dtype=np.uint8)
        # todo: take into account negative X and Y after homography
        # create final image shape
        identity_homography = torch.eye(3, device=self.image1.device)
        homography_matrix = torch.tensor(self.H, device=self.image1.device, dtype=torch.float32)
        self.image1_warp = self.warp_image(self.image1, identity_homography, self.final_image_dimensions)
        self.image2_warp = self.warp_image(self.image2, homography_matrix, self.final_image_dimensions)
        # Warp image 1 with identity and image 2 with homography

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
        if self.debug:
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
        patch_size (tuple): The size of each patch, overlap.
        
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
        patch_width, patch_height, overlap = patch_size
        # get patch width and height

        increment = abs((patch_width - overlap) * intersection_normal[1]) if x_dominant else abs((patch_height - overlap) * intersection_normal[0])
        #increment = patch_width - overlap if x_dominant else patch_height - overlap
        if(x_dominant):
            for i in range(math.ceil((x_max - x_min) / increment)):
                x = x_min + i * increment
                y = y0 + (x - x0) * dy / dx
                patch = (x, y - patch_height / 2, x + patch_width, x_max, y + patch_height / 2)
                patches.append(patch)
        else:
            for i in range(math.ceil((y_max - y_min) / increment)):
                y = y_min + i * increment
                x = x0 + (y - y0) * dx / dy
                patch = (x - patch_width / 2, y, x + patch_width / 2, y + patch_height)
                patches.append(patch)
        # find intersection of intersection area and line given by centroid and normal

        return patches

    def get_padded_patch(self, image, patch):
        """
        Extracts a padded patch from the given image tensor.
        
        Args:
        image (torch.Tensor): The input image tensor of shape [C, H, W].
        patch (tuple): The patch coordinates (x_min, y_min, x_max, y_max).
        
        Returns:
        torch.Tensor: The extracted patch tensor of shape [C, H_patch, W_patch].
        """
        x_min, y_min, x_max, y_max = patch

        pad_left = max(0, -x_min)
        pad_top = max(0, -y_min)
        pad_right = max(0, x_max - image.shape[2])
        pad_bottom = max(0, y_max - image.shape[1])
        # calculate the padding amounts

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        padded_image = torch.nn.functional.pad(image, padding, mode='constant', value=0)
        # pad the image

        x_min += pad_left
        y_min += pad_top
        x_max += pad_left
        y_max += pad_top
        # calculate new patch coordinates in the padded image

        patch_img = padded_image[:, y_min:y_max, x_min:x_max]
        # extract the patch

        return patch_img

    def process_patches(self, patches, intersection_normal, patch_size):
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
            patch2 = [int(min(patch2[0][0], patch2[3][0])), int(min(patch2[0][1], patch2[1][1])), int(min(patch2[0][0], patch2[3][0]))+patch_size[0], int(min(patch2[0][1], patch2[1][1]))+patch_size[1]]
            # get patch2 coordinates

            #patch1_img = self.get_padded_patch(self.image1, patch1)
            #patch2_img = self.get_padded_patch(self.image2, patch2)
            # extract image patches - > approach from original images

            patch1_img = self.get_padded_patch(self.image1_warp, patch1)
            patch2_img = self.get_padded_patch(self.image2_warp, patch1)
            # extract image patches - > approach from transformed images

            patch1_list.append(patch1_img)
            patch2_list.append(patch2_img)

        predicted_flows = self.flow_estimator.process_images(patch1_list, patch2_list, [patch_size[0], patch_size[1]])
        # get optical flow between patches

        center_distance = patch_size[0] - patch_size[2]
        limit = math.sqrt((patch_size[0] / 2) ** 2 - (center_distance / 2) ** 2)
        # compute normalizing factor for the weights using circular mask and overlap value

        for i, (patch1, patch2) in enumerate(zip(patch1_list, patch2_list)):
            flow = predicted_flows[i].squeeze(0).to(patch1.device)  # Flow tensor [2, H, W]

            weight_array = torch.zeros(patch1.shape[1], patch1.shape[2], dtype=torch.float32, device=patch1.device)
            for y in range(weight_array.shape[0]):
                for x in range(weight_array.shape[1]):
                    #weight_array[y, x] = intersection_normal[0] * (x - weight_array.shape[1] / 2 + 0.5) + intersection_normal[1] * (y - weight_array.shape[0] / 2 + 0.5)
                    weight_array[y, x] = min(max((intersection_normal[0] * (x - weight_array.shape[1] / 2 + 0.5) + intersection_normal[1] * (y - weight_array.shape[0] / 2 + 0.5)) / limit, -1), 1)
            # Create weight array

            max_value = weight_array.max()
            min_value = weight_array.min()
            weight_array = (weight_array - min_value) / (max_value - min_value)
            # Normalize weight array

            #weight_np = weight_array.detach().cpu().numpy()
            #plt.imshow(weight_np, cmap='viridis')
            #plt.colorbar()
            #plt.title("Visualization of weight_array")
            #plt.show()

            flow_x = flow[1] * (1 - weight_array)
            flow_y = flow[0] * (1 - weight_array)
            # Modify flow with weight array # !!!!!! opencv uses x, y coordinates, pytorch uses y, x coordinates !!!

            h, w = weight_array.shape
            grid_x, grid_y = torch.meshgrid(torch.arange(h, device=patch1.device), torch.arange(w, device=patch1.device), indexing='ij')
            grid_x = grid_x.float()
            grid_y = grid_y.float()
            # Create meshgrid

            new_x = grid_x + flow_x
            new_y = grid_y + flow_y
            # Calculate new grid positions

            new_x = 2 * new_x / (w - 1) - 1
            new_y = 2 * new_y / (h - 1) - 1
            # Normalize grid positions to the range [-1, 1]

            grid = torch.stack((new_y, new_x), dim=-1).unsqueeze(0)
            # Stack and permute to get grid [H, W, 2]
            
            patch2 = patch2.unsqueeze(0)  # Add batch dimension
            patch2_wrap = F2.grid_sample(patch2, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0)
            # Warp patch2 using grid_sample
            
            if self.debug:
                cv2.imshow('im1', (patch1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                cv2.imshow('im2', (patch2.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                cv2.imshow('composite', (patch2_wrap.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                cv2.waitKey(0)

            valid_mask = (new_x >= -1) & (new_x <= 1) & (new_y >= -1) & (new_y <= 1)
            # Create valid mask

            processed_patches.append((patch2_wrap, valid_mask))

        return processed_patches

    def stitch_images(self, patches, processed_patches, intersection_center, intersection_normal):
        """
        Stitches the processed patches together to form the final stitched image.

        Parameters:
        patches (list): A list of patch positions.
        processed_patches (list): A list of processed patches.

        Returns:
        torch.Tensor: The final stitched image.
        """
        
        self.final_image = create_composite_image(self.image1_warp, self.image2_warp, intersection_center, intersection_normal)
        # Create composite image
        
        for patch, (processed_patch, mask) in zip(patches, processed_patches):
            x_min, y_min, x_max, y_max = patch
            h, w = processed_patch.shape[1:]
            
            x_min = int(x_min)
            y_min = int(y_min)
            
            roi_y1 = max(0, y_min)
            roi_y2 = min(self.final_image.shape[1], y_min + h)
            roi_x1 = max(0, x_min)
            roi_x2 = min(self.final_image.shape[2], x_min + w)
            # Calculate ROI coordinates
            
            patch_y1 = max(0, -y_min)
            patch_y2 = patch_y1 + (roi_y2 - roi_y1)
            patch_x1 = max(0, -x_min)
            patch_x2 = patch_x1 + (roi_x2 - roi_x1)
            # Calculate patch coordinates
            
            patch_transformed = processed_patch[:, patch_y1:patch_y2, patch_x1:patch_x2]
            valid_mask = mask[patch_y1:patch_y2, patch_x1:patch_x2]
            # Extract patch and mask
            
            roi = self.final_image[:, roi_y1:roi_y2, roi_x1:roi_x2]
            # Get the ROI from the final image
            
            valid_mask_expanded = valid_mask.unsqueeze(0).expand_as(patch_transformed)
            roi = torch.where(valid_mask_expanded, patch_transformed, roi)
            # Apply valid mask and blend patches
            
            self.final_image[:, roi_y1:roi_y2, roi_x1:roi_x2] = roi
            # Place the modified ROI back into the final image

            final_image_np = (self.final_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            # Convert final image back to NumPy for visualization if needed

            final_image_np = final_image_np.copy()
            final_image_np = cv2.cvtColor(final_image_np, cv2.COLOR_RGB2BGR)
            cv2.rectangle(final_image_np, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            # Draw recrangle around it
            
            if True:#self.debug:
                cv2.imshow('final', final_image_np)
                cv2.waitKey(0)
            
        return self.final_image

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
        processed_patches = self.process_patches(patches, intersection_normal, patch_size)
        final_image = self.stitch_images(patches, processed_patches, intersection_center, intersection_normal)
        return final_image

    def warp_image(self, image, homography, out_dims):
        """
        Warp image using a homography matrix with PyTorch.

        Args:
        image (torch.Tensor): Image tensor of shape [C, H, W].
        homography (torch.Tensor): Homography matrix of shape [3, 3].
        out_dims (tuple): Dimensions of the output image (H_out, W_out).

        Returns:
        torch.Tensor: Warped image tensor of shape [C, H_out, W_out].
        """
        C, H, W = image.shape
        H_out, W_out = out_dims

        y_out, x_out = torch.meshgrid(torch.linspace(0, H_out - 1, H_out, device=image.device), 
                                    torch.linspace(0, W_out - 1, W_out, device=image.device), indexing='ij')
        ones_out = torch.ones_like(x_out)
        grid_out = torch.stack([x_out, y_out, ones_out], dim=-1).view(-1, 3).t()  # Shape [3, H_out*W_out]
        # create a meshgrid of coordinates in the output image

        inv_homography = torch.inverse(homography)
        original_grid = inv_homography @ grid_out
        original_grid = original_grid[:2] / original_grid[2]  # normalize by the third coordinate
        # apply the inverse homography to the output grid to get coordinates in the original image

        original_grid = original_grid.t().view(H_out, W_out, 2)
        original_grid = original_grid.permute(2, 0, 1).unsqueeze(0)  # Shape [1, 2, H_out, W_out]
        # reshape back to the grid shape [H_out, W_out, 2]

        original_grid[0, 0] = 2 * original_grid[0, 0] / (W - 1) - 1
        original_grid[0, 1] = 2 * original_grid[0, 1] / (H - 1) - 1
        original_grid = original_grid.permute(0, 2, 3, 1)  # Shape [1, H_out, W_out, 2]
        # normalize grid coordinates to [-1, 1]

        image = image.unsqueeze(0)  # Add batch dimension
        warped_image = F2.grid_sample(image, original_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        # warp the image using grid_sample
        
        return warped_image.squeeze(0)
    
def parse_args():
    """
    Dummy function to parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Stitcher")
    parser.add_argument('--img1', required=True, help="Path to the first image")
    parser.add_argument('--img2', required=True, help="Path to the second image")
    #parser.add_argument('--H', required=True, help="Path to the homography matrix in .npy format")
    return parser.parse_args()

if '__main__' == __name__:
    """
    Dummy main function to test the ImageStitcher class.
    Run as python -m libs.image_stitch --img1 path --img2 path from a command line.
    """
    args = parse_args()

    img1 = read_image(args.img1)
    img2 = read_image(args.img2)
    # load images

    from pprint import pprint
    homography = make_homography_with_downscaling(**{  # Expects following argparse arguments.
        'max_size': 1200, # Homography is computed on images of this size
        'device'  : 'cuda',
        'debug'   : False})
    H_data = homography(
                img1,
                img2)

    
    H = np.linalg.inv(H_data[0])
    # load homography matrix
    
    stitcher = ImageStitcher(img1, img2, H, True)
    # initialize the ImageStitcher

    #patch_size = (256, 256, 128)  # Example patch size - width, height, overlap
    patch_size = (192, 192, 96)  # Example patch size - width, height, overlap
    final_image = stitcher.stitch(patch_size)
    # perform the stitching
    
    if final_image is not None:
        cv2.imwrite('stitched_image.jpg', (final_image.detach().cpu().numpy() * 255).astype(np.uint8))
        print("Stitched image saved as 'stitched_image.jpg'.")
    else:
        print("Error: Stitching failed.")
    # save or display the final stitched image
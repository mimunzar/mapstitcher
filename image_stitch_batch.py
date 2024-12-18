import os
import numpy as np
import cv2
import argparse
import math
import gc

import torch
import torch.nn.functional as F2

from libs.optical_flow_raft import OpticalFlow_RAFT
from libs.optical_flow_cv import OpticalFlow_CV
from libs.homography_optimizer import HomographyOptimizer

import torch.nn.functional as Fnn

from libs.util import *
from libs.homography import *

import matplotlib.pyplot as plt

import glymur

class ImageStitcher:
    def __init__(self, subsample=1.0, flow_alg='cv', vram=8.0, debug=False, silent=False):
        #self.optical_flow = OpticalFlow_RAFT()
        self.subsample_flow = subsample
        self.debug = debug
        self.silent = silent
        self.flow_alg = flow_alg
        self.vram = vram

    def find_intersection(self, corners1, corners2, img1_shape, img2_shape):
        """
        Finds the intersection of two images.
        """
        contour1 = np.array(corners1, dtype=np.float32).reshape((-1, 1, 2))
        contour2 = np.array(corners2, dtype=np.float32).reshape((-1, 1, 2))
        # convert corners to contours - cv2

        intersection = cv2.intersectConvexConvex(contour1, contour2)
        if intersection[1] is None:
            return None, None, None, None
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

        if abs(intersection_normal[0]) > abs(intersection_normal[1]):
            y_min = 0
            # max y from all corners
            y_max = max(contour1[:, 0, 1].max(), contour2[:, 0, 1].max())
        else:
            x_min = 0
            #x_max = max(img1_shape[1], img2_shape[1])
            x_max = max(contour1[:, 0, 0].max(), contour2[:, 0, 0].max())

        intersection = intersection_points
        intersection_bb = (x_min, y_min, x_max, y_max)
        intersection_center = intersection_center
        intersection_normal = intersection_normal[0:2]
        
        return intersection, intersection_bb, intersection_center, intersection_normal

    def hard_border_image(self, img, point, normal):
        height, width, _ = img.shape
        y, x = np.meshgrid(np.arange(height).astype(np.float32), np.arange(width).astype(np.float32), indexing='ij')
        
        nx, ny = normal
        px, py = point
        
        mask = (nx * (x - px) + ny * (y - py)) < 0
        mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        border_image = np.where(mask_expanded, img, 0)
        
        return border_image

    def soft_border_image(self, img1, img2, point, normal, mask):
        assert img1.shape == img2.shape, "Images must have the same size"
        
        height, width, _ = img1.shape
        y, x = np.meshgrid(np.arange(height).astype(np.float32), np.arange(width).astype(np.float32), indexing='ij')

        nx, ny = normal
        px, py = point
        #print(nx, ny, px, py)
        
        # Compute the signed distance from the line defined by the point and normal
        signed_distance = nx * (x - px) + ny * (y - py)
        
        # Adjust the range of distances to smoothen the blending
        max_abs_distance = max(np.abs(np.min(signed_distance)), np.abs(np.max(signed_distance)))

        # Normalize the distance to range from 0 (fully img1) to 1 (fully img2)
        blending_mask = (0.5 * (1 + signed_distance / max_abs_distance)) * 2
        blending_mask = np.clip(blending_mask, 0, 1)  # Ensure the mask is between 0 and 2

        # Set 0 where self.mask_overlap is 0
        blending_mask_sin = blending_mask

        # Identify the left side of the line (where signed_distance is negative)
        left_side = signed_distance < 0
        right_side = signed_distance >= 0

        # Square the values of the blending mask on the left side for slower easing
        #blending_mask_sin[left_side] = blending_mask_sin[left_side] ** 2
        #blending_mask_sin[left_side] = 0
        blending_mask_sin = 1.0 - (abs(signed_distance) / max_abs_distance) * 5.0
        blending_mask_sin[right_side] = 1
        blending_mask_sin = np.clip(blending_mask_sin, 0, 1)
        blending_mask_sin = blending_mask_sin * mask

        blending_mask_expanded = np.repeat(blending_mask_sin[:, :, np.newaxis], 3, axis=2)
        #mask_overlap_expanded = np.repeat(self.mask_overlap[:, :, np.newaxis], 3, axis=2)

        #cv2.namedWindow('img', cv2. WINDOW_NORMAL)
        #cv2.resizeWindow('img', 800, 800)
        #cv2.imshow('img', blending_mask_expanded)
        #cv2.waitKey(0)

        soft_image = (img1 * (1 - blending_mask_expanded) + img2 * blending_mask_expanded).astype(np.uint8)

        '''cv2.namedWindow('img', cv2. WINDOW_NORMAL)
        cv2.resizeWindow('img', 1000, 1000)
        while True:
            cv2.imshow('img', img1)
            if cv2.waitKey(0) == ord('q'):
                break

            cv2.imshow('img', img2)
            if cv2.waitKey(0) == ord('q'):
                break

            cv2.imshow('img', soft_image)
            if cv2.waitKey(0) == ord('q'):
                break'''

        return soft_image

    def stitch_set(self, images, Hs):
        min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
        corners_list = []
        image_placed = []

        for i, image in enumerate(images):
            h, w, _ = image.shape
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
            corners = np.dot(np.linalg.inv(Hs[i]), corners)
            corners /= corners[2]
            min_x = min(min_x, corners[0].min())
            min_y = min(min_y, corners[1].min())
            max_x = max(max_x, corners[0].max())
            max_y = max(max_y, corners[1].max())
            image_placed.append(False)

        x_offset = -min_x
        y_offset = -min_y

        H_offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]]).astype(np.float32)

        # transform all homographies to the new coordinate system
        for i in range(len(images)):
            Hs[i] = np.dot(H_offset, np.linalg.inv(Hs[i]))

        # create new corners
        for i, image in enumerate(images):
            h, w, _ = image.shape
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
            corners = np.dot(Hs[i], corners)
            
            corners /= corners[2]
            corners = corners[:2]
            corners_list.append(corners)

        # create canvas
        canvas_width = math.ceil(max_x) - math.floor(min_x)
        canvas_height = math.ceil(max_y) - math.floor(min_y)
        result_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for i in range(len(images)):
            if i == len(images) - 1:
                break

            img1 = images[i]
            corners1 = corners_list[i]
            H1 = Hs[i]
            img2 = images[(i + 1) % len(images)]
            corners2 = corners_list[(i + 1) % len(images)]
            H2 = Hs[(i + 1) % len(images)]

            isect, isect_bb, isect_center, isect_normal = self.find_intersection(corners1.T, corners2.T, img1.shape, img2.shape)

            if isect is None:
                continue

            # warp the images
            image1_warp = cv2.warpPerspective(img1, H1, (canvas_width, canvas_height))
            image2_warp = cv2.warpPerspective(img2, H2, (canvas_width, canvas_height))

            # get the overlapping regions
            overlap1 = image1_warp[int(isect_bb[1]):int(isect_bb[3]), int(isect_bb[0]):int(isect_bb[2])]
            overlap2 = image2_warp[int(isect_bb[1]):int(isect_bb[3]), int(isect_bb[0]):int(isect_bb[2])]

            #cv2.namedWindow('img', cv2. WINDOW_NORMAL)
            #cv2.resizeWindow('img', 1000, 1000)
            #while True:
            #    cv2.imshow('img', overlap1)
            #    if cv2.waitKey(0) == ord('q'):
            #        break

            #    cv2.imshow('img', overlap2)
            #    if cv2.waitKey(0) == ord('q'):
            #        break

            if not image_placed[i]:
                image1_warp = self.hard_border_image(image1_warp, isect_center, isect_normal)
                result_canvas = np.where(image1_warp > 0, image1_warp, result_canvas)
            image2_warp = self.hard_border_image(image2_warp, isect_center, -isect_normal)
            result_canvas = np.where(image2_warp > 0, image2_warp, result_canvas)
            '''cv2.namedWindow('img', cv2. WINDOW_NORMAL)
            cv2.resizeWindow('img', 1000, 1000)
            cv2.imshow('img', result_canvas)
            cv2.waitKey(0)
            continue'''

            mask_full = np.ones((img2.shape[0], img2.shape[1]), dtype=np.float32)
            mask_full = cv2.warpPerspective(mask_full, H2, (canvas_width, canvas_height), flags=cv2.INTER_NEAREST)
            mask_overlap = mask_full[int(isect_bb[1]):int(isect_bb[3]), int(isect_bb[0]):int(isect_bb[2])]

            midpt = [isect_center[0] - isect_bb[0], isect_center[1] - isect_bb[1]]

            # opverlaps to GPU, cuda, tensor
            #print("Before processing:")
            #print(f"Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            #print(f"Reserved: {torch.cuda.memory_reserved() / 1e6} MB")

            if self.flow_alg == 'raft':
                with torch.no_grad():
                    subsample = self.subsample_flow
                    img1_overlap_sub = overlap1
                    img2_overlap_sub = overlap2

                    # subsample flow
                    if subsample > 0.0 and subsample != 1.0:
                        img1_overlap_sub = cv2.resize(overlap1, (int(overlap1.shape[1] / subsample), int(overlap1.shape[0] / subsample)))
                        img2_overlap_sub = cv2.resize(overlap2, (int(overlap2.shape[1] / subsample), int(overlap2.shape[0] / subsample)))
                    
                    # Compute optical flow
                    optical_flow = OpticalFlow_RAFT(silent=self.silent)
                    orientation = 'vertical' if abs(isect_normal[0]) > abs(isect_normal[1]) else 'horizontal'
                    flow = optical_flow.compute_optical_flow_tiled(img1_overlap_sub, img2_overlap_sub, orientation, self.vram)
                    
                    # upsample flow
                    if subsample > 0.0 and subsample != 1.0:
                        flow = cv2.resize(flow, (overlap1.shape[1], overlap1.shape[0]), interpolation=cv2.INTER_NEAREST)
                        flow = flow * subsample

                    flow = flow.transpose(2, 0, 1)
            else:
                # try opencv optical flow
                subsample = self.subsample_flow
                img1_overlap_sub = overlap1
                img2_overlap_sub = overlap2

                # subsample flow
                if subsample > 0.0 and subsample != 1.0:
                    img1_overlap_sub = cv2.resize(overlap1, (int(overlap1.shape[1] / subsample), int(overlap1.shape[0] / subsample)))
                    img2_overlap_sub = cv2.resize(overlap2, (int(overlap2.shape[1] / subsample), int(overlap2.shape[0] / subsample)))

                # compute flow
                optical_flow = OpticalFlow_CV()
                flow = optical_flow.process_images(img1_overlap_sub, img2_overlap_sub)

                # upsample flow
                if subsample > 0.0 and subsample != 1.0:
                    flow = cv2.resize(flow, (overlap1.shape[1], overlap1.shape[0]), interpolation=cv2.INTER_NEAREST)
                    flow = flow * subsample
                
                flow = flow.transpose(2, 0, 1)

            img_mid, mask_overlap = self.remap_image_with_flow(overlap2, flow, isect_normal, midpt, mask_overlap)
            #img_mid = (img_mid.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            #print("After processing:")
            #print(f"Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            #print(f"Reserved: {torch.cuda.memory_reserved() / 1e6} MB")

            #print(torch.cuda.memory_summary(device='cuda:0', abbreviated=False))

            # blend the images
            overlap = result_canvas[int(isect_bb[1]):int(isect_bb[3]), int(isect_bb[0]):int(isect_bb[2])]
            result_canvas[int(isect_bb[1]):int(isect_bb[3]), int(isect_bb[0]):int(isect_bb[2])] = self.soft_border_image(overlap, img_mid, midpt, isect_normal, mask_overlap)

            image_placed[i] = True
            image_placed[(i + 1) % len(images)] = True

        return result_canvas, H_offset

    def remap_image_with_flow(self, img2_overlap, flow, intersection_normal, point, mask): # todo: for arbitrary image position needs to be changed
        #img2_overlap_np = img2_overlap.squeeze(0).cpu().numpy()  # Shape: [C, H, W]
        img2_overlap_np = img2_overlap  # Shape: [H, W, C]
        flow_np = flow#flow.squeeze(0).cpu().numpy()  # Shape: [2, H, W]
        
        #img2_overlap_np = img2_overlap_np.transpose(1, 2, 0)  # Shape: [H, W, C]
        flow_np = flow_np.transpose(1, 2, 0)  # Shape: [H, W, 2]
        
        h, w, _ = img2_overlap_np.shape
        flow_np_resized = flow_np#cv2.resize(flow_np, (w, h))  # Resize flow if needed

        y, x = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing='ij')

        nx, ny = intersection_normal
        px, py = point
        
        # Compute the signed distance from the line defined by the point and normal
        signed_distance = nx * (x - px) + ny * (y - py)
        
        # Adjust the range of distances to smoothen the blending
        max_abs_distance = min(min(np.abs(signed_distance[0, 0]), np.abs(signed_distance[-1, -1])), min(np.abs(signed_distance[0, -1]), np.abs(signed_distance[-1, 0])))
        max_abs_distance = max_abs_distance * 0.9 # avoid (right) border artifacts

        # Normalize the distance to range from 0 (fully img1) to 1 (fully img2)
        blending_mask = (1.0 - (0.5 * (1 + signed_distance / max_abs_distance))) * 2.0
        blending_mask = np.clip(blending_mask, 0, 1)  # Ensure the mask is between 0 and 2

        map_x, map_y = np.meshgrid(np.arange(w).astype(np.float32), np.arange(h).astype(np.float32))
        map_x = map_x + flow_np_resized[:, :, 0] * blending_mask
        map_y = map_y + flow_np_resized[:, :, 1] * blending_mask

        img2_remapped = cv2.remap(img2_overlap_np, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
        mask = cv2.remap(mask, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_NEAREST)

        #cv2.namedWindow('img', cv2. WINDOW_NORMAL)
        #cv2.resizeWindow('img', 800, 800)
        #while True:
        #    cv2.imshow('img', img2_overlap_np)
        #    if cv2.waitKey(0) == ord('q'):
        #        break

        #    cv2.imshow('img', img2_remapped)
        #    if cv2.waitKey(0) == ord('q'):
        #        break
            
        #    cv2.imshow('img', mask)
        #    if cv2.waitKey(0) == ord('q'):
        #        break
        
        #img2_remapped_tensor_np = (img2_remapped * 255).astype(np.uint8)
        img2_remapped_tensor_np = img2_remapped
        
        return img2_remapped_tensor_np, mask

def parse_list_file(config_file, silent=False):
    images = []
    rows = []

    # Extract the base path from config_file
    base_path = os.path.dirname(config_file)

    if not silent:
        print(f"Loading config file: {config_file}")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")

    with open(config_file, 'r') as f:
        lines = f.readlines()

    section = None  # None initially, then 'images' or 'rows'

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue

        if line.startswith('-'):
            # Determine the section based on the current line
            if 'images' in line:
                section = 'images'
            elif 'rows' in line:
                section = 'rows'
            continue

        # Parse based on the current section
        if section == 'images':
            index, path = line.split(maxsplit=1)
            # Append the full path
            full_path = os.path.join(base_path, path)
            images.append(full_path)
        elif section == 'rows':
            indices = tuple(map(int, line.split()))
            rows.append(indices)

    return images, rows

def get_all_neighbours(rows):
    rows = np.array(rows)  # Convert to NumPy array for easier manipulation
    num_rows, num_cols = rows.shape

    # Flatten indices and find valid (non-'X') elements
    valid_indices = [(i, j) for i in range(num_rows) for j in range(num_cols) if rows[i, j] != 'X']
    
    # Find the approximated middle index
    mid_row = num_rows // 2
    mid_col = num_cols // 2
    # Find the closest valid index to the middle
    mid_index = min(
        valid_indices,
        key=lambda idx: abs(idx[0] - mid_row) + abs(idx[1] - mid_col)
    )
    start_index = rows[mid_index]  # The value at the middle index

    # Create pairs of neighbors
    pairs = []
    for i, j in valid_indices:
        current = rows[i, j]
        # Check all neighbors of the current element
        neighbors = [
            (i, j - 1),  # Left
            (i, j + 1),  # Right
            (i - 1, j),  # Up
            (i + 1, j),  # Down
        ]
        for ni, nj in neighbors:
            if 0 <= ni < num_rows and 0 <= nj < num_cols and rows[ni, nj] != 'X':
                neighbor = rows[ni, nj]
                if (current, neighbor) not in pairs and (neighbor, current) not in pairs:
                    pairs.append((current, neighbor))
    
    # Sort pairs to prioritize connections involving the start index
    pairs.sort(key=lambda x: (x[0] != start_index, x[1] != start_index))
    converted_data = [(int(a), int(b)) for a, b in pairs]

    return converted_data

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Stitcher")
    parser.add_argument('--list', required=True, help="File and processing list")
    parser.add_argument('--optimization-model', required=False, default='affine', help="Optimization model homography or affine")
    parser.add_argument('--matching-algorithm', required=False, default='sift', help="Matching algorithm loftr or sift")
    parser.add_argument('--loftr-model', required=False, default='outdoor', help="Model for LOFTR - outdoor or indoor")
    parser.add_argument('--flow-alg', required=False, default='raft', help="Optical Flow algorithm cv or raft")
    parser.add_argument('--subsample-flow', default=2.0, type=float, help="Subsample flow")
    parser.add_argument('--vram-size', default=8.0, type=float, help="GPU VRAM size in GB")
    parser.add_argument('--max-matches', default=800, type=int, help="Maximum number of matches per image pair (for optimization)")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--output', default='result.jp2', help="Output file")
    parser.add_argument('--silent', action='store_true', help="Silent mode")
    return parser.parse_args()

if '__main__' == __name__:
    """
    Main function.
    """
    args = parse_args()
    max_matches = args.max_matches

    images, rows = parse_list_file(args.list, args.silent)
    h_pairs = get_all_neighbours(rows)

    # Output for debugging
    if not args.silent:
        print("Images List:", images)
        print("Pairs List:", h_pairs)
        print("Rows List:", rows)
        print("Optimization Model:", args.optimization_model)
        print("Matching Algorithm:", args.matching_algorithm)
        print("LOFTR Model:", args.loftr_model)
        print("Flow Algorithm:", args.flow_alg)
        print("Subsample Flow:", args.subsample_flow)
    
    # create list of homographies
    if args.matching_algorithm == 'loftr':
        side_size = (int)(math.sqrt((1000 * 1000) * (args.vram_size / 8.0)))
    else:
        side_size = 1500
    if args.optimization_model == 'homography':
        if not args.silent:
            print("Homography size:", side_size)
        transform = make_homography_with_downscaling(**{  # Expects following argparse arguments.
            'max_size': side_size,
            'device'  : 'cuda',
            'model'   : args.loftr_model,
            'algorithm': args.matching_algorithm,
            'debug'   : False})
    else:
        if not args.silent:
            print("Affine size:", side_size)
        transform = make_affine_transform_with_downscaling(**{  # Expects following argparse arguments.
            'max_size': side_size,
            'device'  : 'cuda',
            'model'   : args.loftr_model,
            'algorithm': args.matching_algorithm,
            'debug'   : False})
    
    # this is a workaround to avoid CUDA memory allocation errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    homographies = []
    for image in images:
        homographies.append(np.eye(3))
    corresponding_points = []
    
    image_in = [h_pairs[0][0]]
    connectred_list = []
    # create connected pairs list
    while h_pairs != []:
        pair = h_pairs.pop(0)
        if pair[0] in image_in:
            connectred_list.append(pair)
            image_in.append(pair[1])
        elif pair[1] in image_in:
            connectred_list.append([pair[1], pair[0]])
            image_in.append(pair[0])
        else:
            h_pairs.append(pair)
    print("Connected List:", connectred_list)

    for i, pair in enumerate(connectred_list):
        H_data = transform(read_image(images[pair[0]]), read_image(images[pair[1]]))
        homographies[pair[1]] = np.dot(homographies[pair[0]], H_data[0])
        # update homographies
        # store
        corresponding_points.append({
            'pair': pair,
            'points': H_data[1]
        })

    optimizer = HomographyOptimizer(max_matches, args.optimization_model, silent=args.silent)
    h_optimized = optimizer.optimize(connectred_list, homographies, corresponding_points)
    #h_optimized = homographies

    #homographies = []
    #for image in images:
    #    homographies.append(np.eye(3))
    
    #for i, pair in enumerate(h_pairs):
    #    homographies[pair[1]] = np.dot(homographies[pair[0]], h_optimized[i])

    images_loaded = []
    for image in images:
        images_loaded.append(cv2.imread(image))
    # create result image based on homographies

    image_stitcher = ImageStitcher(args.subsample_flow, args.flow_alg, args.vram_size, args.debug, silent=args.silent)

    rows_canvi = []
    rows_offset = []
    for j, row in enumerate(rows):
        row_images = [images_loaded[i] for i in row]
        row_homographies = [h_optimized[i] for i in row]
        row_canvas, H_offset = image_stitcher.stitch_set(row_images, row_homographies)    
        rows_canvi.append(row_canvas)
        rows_offset.append(H_offset)

    result_canvas = None
    if(len(rows_canvi) == 1):
        result_canvas = rows_canvi[0]
        H_offset = rows_offset[0]
    else:
        result_canvas, H_offset = image_stitcher.stitch_set(rows_canvi, rows_offset)

    # save result
    if not args.silent:
        print("Saving result to", args.output)

    # check suffix of the output file
    suffix = args.output.split('.')[-1]
    if suffix == 'jp2':
        # write lossless jp2
        result_canvas = cv2.cvtColor(result_canvas, cv2.COLOR_BGR2RGB)
        if result_canvas.dtype != np.uint8:
            result_canvas = (result_canvas / 256).astype(np.uint8)
        glymur.Jp2k(args.output, data=result_canvas, psnr=[0])
    else:
        cv2.imwrite(args.output, result_canvas)

import sys
sys.path.append('core')

import configparser
import argparse
import os
import cv2
import glob
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import kornia as K
import kornia.feature as KF
#from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt

from image_map import ImageData
from image_map import ImageMap

#from raft import RAFT
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
# from utils import flow_viz
# from utils.utils import InputPadder

DEVICE = 'cuda'

def get_files(directory, wildcard, sort = True):
    pattern = os.path.join(directory, wildcard)
    files = glob.glob(pattern)
    
    if sort:
        files.sort()

    return files

def img_CV_Color_to_GPU(img):
        img = np.array(img).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

def get_intersection(bb1, bb2):
    x1_min, y1_min, x1_max, y1_max = bb1
    x2_min, y2_min, x2_max, y2_max = bb2
    # Unpack the bounding boxes

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    # Compute the coordinates of the intersection

    if x_inter_min < x_inter_max and y_inter_min < y_inter_max:
        intersection = (x_inter_min, y_inter_min, x_inter_max, y_inter_max)
        return intersection
    # There is an intersection
    else:
        return None
    # No intersection

def match_images_LoFTR(img0, img1, draw_matches = True):
    matcher = KF.LoFTR(pretrained="outdoor")
    # create the LoFTR matcher

    img0_color = img0.get_image('cv')
    img1_color = img1.get_image('cv')
    img0_gpu = img0.get_image('gpu-gray')
    img1_gpu = img1.get_image('gpu-gray')

    print('Img0 shape:', img0_gpu.shape)
    print('Img1 shape:', img1_gpu.shape)

    input_dict = {
        "image0": img0_gpu,
        "image1": img1_gpu,
    }
    # prepare the input dictionary

    with torch.inference_mode():
        correspondences = matcher(input_dict)
    # matching

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    print('Keypoints img0:', len(mkpts0))
    print('Keypoints img1:', len(mkpts1))
    # get the keypoints

    H, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 4.0)
    print('Homography:\n', H)
    # CV homography

    if draw_matches:
        h, w, _   =  [min(x) for x in zip(img0_color.shape, img1_color.shape)]
        composite = np.hstack([
            cv2.resize(img0_color, [w, h]),
            cv2.resize(img1_color, [w, h])])
        for i, color in enumerate([[0, 0, 255], [0, 255, 0]]):
            points0 = mkpts0[i == inlier_mask.flatten()].astype(int)
            points1 = mkpts1[i == inlier_mask.flatten()].astype(int) + [w, 0]
            for x, y in zip(points0, points1):
                cv2.circle(composite, x, 2, color, -1)
                cv2.circle(composite, y, 2, color, -1)
                cv2.line  (composite, x, y, color,  1)
        cv2.imshow('Matches between Images', composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return H


def create_superimage(images, homographies, blending = 0.5):
    # Find the size of the output image
    minX = [0]
    minY = [0]
    maxX = [0]
    maxY = [0]
    for i in range(len(images)):
        pt_lt = homographies[i] @ np.array([0, 0, 1]).reshape(-1, 1)# [[0], [0], [1]]
        pt_lt = pt_lt / pt_lt[2]
        pt_rb = homographies[i] @ np.array([images[i].shape[1], images[i].shape[0], 1]).reshape(-1, 1) #[[images[i].shape[1]], [images[i].shape[0]], [1]]
        pt_rb = pt_rb / pt_rb[2]

        minX = min(minX, pt_lt[0])
        minY = min(minY, pt_lt[1])
        maxX = max(maxX, pt_rb[0])
        maxY = max(maxY, pt_rb[1])

    # Create a large canvas to accommodate the superimage
    superimage_height = int(maxY[0] - minY[0])
    superimage_width = int(maxX[0] - minX[0])
    superimage = np.zeros((superimage_height, superimage_width, 3), dtype=np.uint8)
    offsetX = -minX[0]
    offsetY = -minY[0]

    for i in range(len(images)):
        img = images[i]
        H = homographies[i]
        H[0, 2] += offsetX
        H[1, 2] += offsetY
        homographies[i] = H # update offset
        # Warp the image using the homography
        warped_img = cv2.warpPerspective(img, H, (superimage_width, superimage_height))

        # Create a mask for the current image
        mask = cv2.warpPerspective(np.ones((img.shape[0], img.shape[1]), dtype=np.uint8), H, (superimage_width, superimage_height))

        # Blend the current image into the superimage
        superimage[mask == 1] = warped_img[mask == 1] * (1 - blending) + superimage[mask == 1] * blending

    return superimage

def process(args):
    config_path = get_files(args.image_data, "*.ini")[0]
    config = configparser.ConfigParser()
    config.read(config_path)
    # get configuration

    model = raft_large(pretrained=True)
    model = model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #model = torch.nn.DataParallel(RAFT(args))
    #model.load_state_dict(torch.load(args.model))
    #model = model.module
    #model.to(DEVICE)
    #model.eval()
    # set up the model

    image_map = ImageMap()
    # create the image map

    image_list = get_files(args.image_data, "*.jpg")
    # get the list of images

    image_count = len(image_list) if config.getint('LoFTR', 'image_count') == -1 else config.getint('LoFTR', 'image_count')
    for i in range(len(image_list)):
        image_map.load_image(image_list[i], config.getint('LoFTR', 'subsample'))
        if(image_count != -1 and i >= image_count - 1):
            break

    for i in range(image_map.get_image_count() - 1):
        img0 = image_map.get_image_data()[i]
        img1 = image_map.get_image_data()[i + 1]
        # load images

        H = match_images_LoFTR(img0, img1, True)
        H = np.linalg.inv(H)

        # match images
        img1.H = np.matmul(H, img0.H)
        # update homography
    # todo: smarter way to match images
    # todo: oprimize homographies using graph optimization

    images = [img.get_image('cv') for img in image_map.get_image_data()]
    homographies = [img.H for img in image_map.get_image_data()]
    # get images and homographies
    superimage = create_superimage(images, homographies)
    superimage_result = create_superimage(images, homographies, 0.0)
    # draw images to one big image according to homographies

    flow = np.zeros((superimage.shape[0], superimage.shape[1], 2), dtype=np.float32)
    # create optical flow image in the size of superimage

    for i in range(image_count - 1):
        bb0_pts = np.array([[0, 0, 1], [images[i].shape[1], 0, 1], [images[i].shape[1], images[i].shape[0], 1], [0, images[i].shape[0], 1]]) # homogeneous coordinates
        bb1_pts = np.array([[0, 0, 1], [images[i + 1].shape[1], 0, 1], [images[i + 1].shape[1], images[i + 1].shape[0], 1], [0, images[i + 1].shape[0], 1]]) # homogeneous coordinates

        bb0_pts = bb0_pts @ homographies[i].T
        bb0_pts = bb0_pts / bb0_pts[:, 2].reshape(-1, 1)
        bb1_pts = bb1_pts @ homographies[i + 1].T
        bb1_pts = bb1_pts / bb1_pts[:, 2].reshape(-1, 1)
        # transform bounding boxes

        blend_vector = np.array([bb1_pts[0][0] - bb0_pts[0][0], bb1_pts[0][1] - bb0_pts[0][1]])
        norm = np.linalg.norm(blend_vector)
        blend_vector = blend_vector / norm if norm != 0 else blend_vector
        # get blend vector

        bb0_rect = np.array([min(bb0_pts[:, 0]), min(bb0_pts[:, 1]), max(bb0_pts[:, 0]), max(bb0_pts[:, 1])])
        bb1_rect = np.array([min(bb1_pts[:, 0]), min(bb1_pts[:, 1]), max(bb1_pts[:, 0]), max(bb1_pts[:, 1])])
        # get minX, minY, maxX, maxY

        intersection = get_intersection(bb0_rect, bb1_rect)

        if intersection is not None:
            cv2.rectangle(superimage, (int(intersection[0]), int(intersection[1])), (int(intersection[2]), int(intersection[3])), (0, 255, 0), 3)
        # draw intersecting areas

        img_warp0 = cv2.warpPerspective(images[i], homographies[i], (superimage.shape[1], superimage.shape[0]))
        subimage0 = img_warp0[int(intersection[1]):int(intersection[3]), int(intersection[0]):int(intersection[2])]
        img_warp1 = cv2.warpPerspective(images[i + 1], homographies[i + 1], (superimage.shape[1], superimage.shape[0]))
        subimage1 = img_warp1[int(intersection[1]):int(intersection[3]), int(intersection[0]):int(intersection[2])]
        # extract sub images

        cv2.imshow('Sub0', subimage0)
        cv2.imshow('Sub1', subimage1)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        subimage0_gpu = img_CV_Color_to_GPU(subimage0)
        subimage1_gpu = img_CV_Color_to_GPU(subimage1)
        padder = InputPadder(subimage0_gpu.shape)
        subimage0_gpu, subimage1_gpu = padder.pad(subimage0_gpu, subimage1_gpu)
        # pad the images

        with torch.no_grad():
            flow_up = model(subimage0_gpu, subimage1_gpu)[-1]
        #flow_low, flow_up = model(subimage0_gpu, subimage1_gpu)
        flow_np = flow_up.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        #flow_np = flow_to_image(flow_up).cpu().numpy()
        # compute optical flow

        weight_array = np.zeros((subimage0.shape[0], subimage0.shape[1]), dtype=np.float32)
        for y in range(weight_array.shape[0]):
            for x in range(weight_array.shape[1]):
                weight_array[y, x] = blend_vector[0] * (x - weight_array.shape[1] / 2) + blend_vector[1] * (y - weight_array.shape[0] / 2)
        # create weight array
        max_value = np.max(weight_array)
        min_value = np.min(weight_array)
        weight_array = (weight_array - min_value) / (max_value - min_value)
        # normalize weight array
        flow_x = flow_np[..., 0]
        flow_y = flow_np[..., 1]

        x, y = np.meshgrid(np.arange(flow_np.shape[1]), np.arange(flow_np.shape[0]))
        new_x = x + flow_x
        new_y = y + flow_y
        subimage1_warp = cv2.remap(subimage1, new_x.astype(np.float32), new_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
        # create warped image

        blended = np.array(subimage0)
        for y in range(weight_array.shape[0]):
            for x in range(weight_array.shape[1]):
                blended[y, x] = subimage0[y, x] * (1 - weight_array[y, x]) + subimage1_warp[y, x] * weight_array[y, x]
        # create blended image

        superimage_result[int(intersection[1]):int(intersection[3]), int(intersection[0]):int(intersection[2])] = blended
        # store to superimage result

        cv2.imshow('Blended', blended)
        cv2.imshow('Superimage', superimage)
        cv2.imshow('Superimage Result', superimage_result)
        cv2.waitKey(0)
        # visualize the blended image

    # find intersecting areas for images

    cv2.imshow('Superimage', superimage)
    cv2.imshow('Superimage Result', superimage_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    #parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    #parser.add_argument('--image_from', help="image0")
    #parser.add_argument('--image_to', help="image1")

    parser.add_argument('--image_data', help="dataset path")

    args = parser.parse_args()

    process(args)


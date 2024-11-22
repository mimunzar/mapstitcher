import numpy as np

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt
import torch.nn.functional as Fnn

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [Fnn.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class OpticalFlow_RAFT:
    def __init__(self, debug=False, silent=False):
        """
        Initializes the OpticalFlow_RAFT class.
        """

        self.silent = silent
        self.debug = debug
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights)
        self.model = self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        # load the RAFT model

    def compute_optical_flow(self, img1_overlap, img2_overlap, subsample=1.0):
        if subsample < 0:
            # auto compute max possible subsample
            maxpix = 1200 * 1200
            subsample = np.sqrt((img1_overlap.shape[2] * img1_overlap.shape[3]) / maxpix)
            if not self.silent:
                print(f"Auto computed subsample: {subsample}")

        if subsample != 1.0:
            img1_overlap_sub = Fnn.interpolate(img1_overlap, scale_factor=1.0/subsample, mode='bilinear', align_corners=False)
            img2_overlap_sub = Fnn.interpolate(img2_overlap, scale_factor=1.0/subsample, mode='bilinear', align_corners=False)
        else:
            img1_overlap_sub, img2_overlap_sub = img1_overlap, img2_overlap

        # Apply padding to subsampled images
        padder = InputPadder(img1_overlap_sub.shape)
        img1_overlap_sub, img2_overlap_sub = padder.pad(img1_overlap_sub, img2_overlap_sub)

        #print(img1_overlap_sub.shape, img2_overlap_sub.shape)
        
        # Compute optical flow on the subsampled images
        with torch.no_grad():
            flow_up_sub = self.model(img1_overlap_sub, img2_overlap_sub)
            # Remove padding from the subsampled flow
            flow_up_unp_sub = padder.unpad(flow_up_sub[0])

            # Upsample the flow back to the original resolution
            # print(flow_up_unp_sub.shape, img1_overlap.shape)
            flow_up_unp = Fnn.interpolate(flow_up_unp_sub, size=(img1_overlap.shape[2], img1_overlap.shape[3]), mode='nearest')
            
            if subsample != 1.0:
                flow_up_unp = flow_up_unp * (subsample)
            
        return flow_up_unp

    def process_image(self, image_list1, image_list2, size):
        """
        Processes the images to extract the optical flow.
        """

        img1_batch = torch.stack(image_list1)
        img2_batch = torch.stack(image_list2)
        # convert images to pytorch tensors

        img1_batch = F.resize(img1_batch, size, antialias=False)
        img2_batch = F.resize(img2_batch, size, antialias=False)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)
        # preprocess the images

        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))

        flow_tensors = [flow for flow in list_of_flows[-1]]
        flow_batch = torch.stack(flow_tensors)

        if self.debug:
            flow_imgs = flow_to_image(flow_batch)
            img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
            grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            self.plot(grid)

        return flow_batch
    
    def plot(self, imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.show()

    def process_images(self, image_list1, image_list2, size):
        """
        Processes the images to extract the optical flow.
        """

        img1_batch = torch.stack(image_list1)
        img2_batch = torch.stack(image_list2)
        # convert images to pytorch tensors

        img1_batch = F.resize(img1_batch, size, antialias=False)
        img2_batch = F.resize(img2_batch, size, antialias=False)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)
        # preprocess the images, todo: patch size

        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))

        flow_tensors = [flow for flow in list_of_flows[-1]]
        flow_batch = torch.stack(flow_tensors)

        if self.debug:
            flow_imgs = flow_to_image(flow_batch)
            img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
            grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            self.plot(grid)

        return flow_batch
    
    def plot(self, imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.show()
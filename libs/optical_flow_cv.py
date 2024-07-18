import cv2
import numpy as np
import torch
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt

class OpticalFlow_CV:
    debug = False

    def __init__(self):
        """
        Initializes the OpticalFlow_CV class.
        """

        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # set the parameters for the optical flow
        self.prev_gray = None
        # initialize the previous frame

    def process_images(self, images0, images1):
        """
        Compute dense optical flow between pairs of images using the Farneback method.
        
        Args:
        images0 (list of np.ndarray): List of first images.
        images1 (list of np.ndarray): List of second images.
        
        Returns:
        list of np.ndarray: List of dense optical flow results.
        """
        assert len(images0) == len(images1), "Both lists of images must have the same length"
        
        optical_flows = []
        
        for img0, img1 in zip(images0, images1):
            # Convert images to grayscale
            gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            
            # Compute dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, 
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            optical_flows.append(flow)

        flow_tensors = [torch.from_numpy(flow).permute(2, 0, 1) for flow in optical_flows]
        flow_batch = torch.stack(flow_tensors)

        if self.debug:
            flow_imgs = flow_to_image(flow_batch)
            patch1_list_py = [torch.from_numpy(img).permute(2, 0, 1).float() for img in images0]
            img1_batch = torch.stack(patch1_list_py)
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
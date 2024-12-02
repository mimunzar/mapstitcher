import cv2
import numpy as np
import torch
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt

class OpticalFlow_CV:
    def __init__(self, debug=False):
        """
        Initializes the OpticalFlow_CV class.
        """

        self.debug = debug
        self.flow_params = dict(pyr_scale=0.5, levels=5, winsize=31, iterations=7, poly_n=7, poly_sigma=1.5, flags=0)
        # set the parameters for the optical flow

    def process_images(self, img0, img1):
        """
        Compute dense optical flow between pairs of images using the Farneback method.
        
        Args:
        images0 (list of np.ndarray): List of first images.
        images1 (list of np.ndarray): List of second images.
        
        Returns:
        list of np.ndarray: List of dense optical flow results.
        """
        assert img0.shape == img1.shape, "Images must have the same shape."

        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 

        '''flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 
                                            self.flow_params.get('pyr_scale'),
                                            self.flow_params.get('levels'),
                                            self.flow_params.get('winsize'),
                                            self.flow_params.get('iterations'),
                                            self.flow_params.get('poly_n'),
                                            self.flow_params.get('poly_sigma'),
                                            self.flow_params.get('flags'))'''
        dis = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        dis.setFinestScale(1)  # Larger scale for document stitching
        dis.setPatchSize(25)   # Larger patches for smooth flow
        dis.setGradientDescentIterations(50)  # Higher iterations for stability
        flow = dis.calc(gray0, gray1, None)
        
        return flow

        '''optical_flows = []
        
        for img0, img1 in zip(images0, images1):
            # Convert images to grayscale
            gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # Compute dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, 
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            #print(max(flow.flatten()), min(flow.flatten()))
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
        
        return optical_flows#flow_batch'''
    
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
import numpy as np

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt

class OpticalFlow_RAFT:
    def __init__(self, debug=False):
        """
        Initializes the OpticalFlow_RAFT class.
        """

        self.debug = debug
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights)
        self.model = self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        # load the RAFT model

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
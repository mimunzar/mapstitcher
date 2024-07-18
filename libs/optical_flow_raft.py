import numpy as np

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt

class OpticalFlow_RAFT:
    debug = True

    def __init__(self):
        """
        Initializes the OpticalFlow_RAFT class.
        """

        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights)
        self.model = self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        # load the RAFT model

    def process_images(self, image_list1, image_list2):
        """
        Processes the images to extract the optical flow.
        """

        patch1_list_py = [torch.from_numpy(img).permute(2, 0, 1).float() for img in image_list1]
        patch2_list_py = [torch.from_numpy(img).permute(2, 0, 1).float() for img in image_list2]
        img1_batch = torch.stack(patch1_list_py)
        img2_batch = torch.stack(patch2_list_py)
        # convert images to pytorch tensors

        img1_batch = F.resize(img1_batch, size=[256, 256], antialias=False)
        img2_batch = F.resize(img2_batch, size=[256, 256], antialias=False)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)
        # preprocess the images, todo: patch size

        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))

        predicted_flows = list_of_flows[-1]

        if self.debug:
            flow_imgs = flow_to_image(predicted_flows)
            img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
            grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            self.plot(grid)

        return predicted_flows
    
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
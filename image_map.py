import cv2
import glob
import numpy as np
import torch

class ImageData:
    def __init__(self, path, image_cv, image_cv_gray, image_gpu, image_gpu_gray):
        self.path = ""
        self.images = {
            'cv': image_cv,
            'cv-gray': image_cv_gray,
            'gpu': image_gpu,
            'gpu-gray': image_gpu_gray
        }
        self.H = np.eye(3)
        # 3x3 homography matrix

    def get_image(self, format):
        return self.images.get(format)

class ImageMap:
    def __init__(self):
        self.image_data = []
        self.DEVICE = 'cuda'

    def load_image(self, path, subsample=1):
        img = cv2.imread(path)
        # cv color
        if subsample > 1:
            img = cv2.resize(img, (img.shape[1] // subsample, img.shape[0] // subsample))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv gray
        img_gpu = (torch.from_numpy(img).permute(2, 0, 1).float())[None].to(self.DEVICE)
        # gpu color
        img_gpu_gray = torch.from_numpy(img_gray)[None][None] / 255.
        # gpu gray

        self.image_data.append(ImageData(path, img, img_gray, img_gpu, img_gpu_gray))

    def get_images_by_format(self, format):
        return [image_data.get_image(format) for image_data in self.image_data]
    
    def get_image_data(self):
        return self.image_data
    
    def get_image_count(self):
        return len(self.image_data)
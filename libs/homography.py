from libs.util import *
import torch


def make_correspondences(device, debug):
    from kornia.feature                    import LoFTR # type: ignore
    from torchvision.transforms.functional import rgb_to_grayscale
    loftr = LoFTR(pretrained = 'outdoor').to(device)
    loftr.eval()
    def correspondences(image0, image1):
        with torch.inference_mode():
            matches = loftr({ # The images have to fit on GPU here
                'image0': rgb_to_grayscale(image0.unsqueeze(0)).to(device),
                'image1': rgb_to_grayscale(image1.unsqueeze(0)).to(device)})
            return matches['confidence'].cpu().numpy(),[
                    matches['keypoints0'].cpu().numpy(), matches['keypoints1'].cpu().numpy()]
    if debug: # Plot side to side correspondences
          from matplotlib import pyplot
          def correspondences_debug(image0, image1):
              conf, [kp0, kp1]  = correspondences(image0, image1)
              fig , [ax0, ax1]  = pyplot.subplots(1, 2, tight_layout = True)
              line_between_axes = make_line_between_axes(fig, ax0, ax1)
              for p0, p1, c in zip(kp0, kp1, conf):
                  line_between_axes(p0, p1, color = [1, 0, 0, float(c)]) # Set alpha based on confidence
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Correspondences 0')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Correspondences 1')
              pyplot.show()
              return conf, [kp0, kp1]
          return correspondences_debug
    else: return correspondences


def make_homography(device, debug):
    from cv2 import findHomography, RANSAC
    correspondences = make_correspondences(device, debug)
    def homography(image0, image1):
        _, [kp0, kp1]  = correspondences(image0, image1)
        H, inlier_mask = findHomography(kp0, kp1, RANSAC, 4.0)
        inlier_mask    = inlier_mask.flatten()
        return H, [kp0[1 == inlier_mask], kp1[1 == inlier_mask]]
    if debug:
          from matplotlib import pyplot
          def homography_debug(image0, image1):
              H  , [kp0, kp1]      = homography(image0, image1)
              fig, [ax0, ax1, ax2] = pyplot.subplots(1, 3, tight_layout = True)
              line_between_axes    = make_line_between_axes(fig, ax0, ax1)
              for p0, p1 in zip(kp0, kp1):
                  line_between_axes(p0, p1, color = [1, 0, 0])
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Inliers 0')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Inliers 1')
              ax2.imshow(warp_image(image0, H)) ; ax2.set_title('Warped')
              pyplot.show()
              return H, [kp0, kp1]
          return homography_debug
    else: return homography


def make_homography_with_downscaling(max_size, device, debug):
    """Computes homography on downscaled images to allow compute feature
    correspondences on GPU with large images. The computed homography is
    corrected to describe perspective transformation on original images.  """
    from numpy import array
    resize_image = make_resize_image(max_size)
    homography   = make_homography(device, debug)
    def homography_with_downscaling(image0, image1):
        small0, scale0     = resize_image(image0)
        small1, scale1     = resize_image(image1)
        H     , [kp0, kp1] = homography(small0, small1)
        #inv_scale0         = 1 / scale0
        #return [H * array([[     1,      1, inv_scale0],
        #                   [     1,      1, inv_scale0],
        #                   [scale0, scale0,          1]]),
        #        [kp0 * inv_scale0, kp1 * 1 / scale1]]
        scale_matrix0 = array([[1 / scale0, 0,         0],
                                [0,         1 / scale0, 0],
                                [0,         0,         1]])
        scale_matrix1 = array([[scale1, 0,      0],
                                [0,      scale1, 0],
                                [0,      0,      1]])
        H_corrected = scale_matrix0 @ H @ scale_matrix1
        return [H_corrected, [kp0 * (1 / scale0), kp1 * (1 / scale1)]]
    if debug:
          from matplotlib import pyplot
          def homography_with_downscaling_debug(image0, image1):
              H  , [kp0, kp1]      = homography_with_downscaling(image0, image1)
              fig, [ax0, ax1, ax2] = pyplot.subplots(1, 3, tight_layout = True)
              line_between_axes    = make_line_between_axes(fig, ax0, ax1)
              for p0, p1 in zip(kp0, kp1):
                  line_between_axes(p0, p1, color = [1, 0, 0])
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Image 0')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Image 1')
              ax2.imshow(warp_image(image0, H)) ; ax2.set_title('Warped')
              pyplot.show()
              return H, [kp0, kp1]
          return homography_with_downscaling_debug
    else: return homography_with_downscaling


if '__main__' == __name__:
    """Run as python -m libs.homography from a command line."""
    from pprint import pprint
    homography = make_homography_with_downscaling(**{  # Expects following argparse arguments.
        'max_size': 128, # Homography is computed on images of this size
        'device'  : 'cuda',
        'debug'   : True})
    pprint(homography(
        read_image('test_data/2610465282_cut_part_1.tif.jpg'),
        read_image('test_data/2610465282_cut_part_2.tif.jpg')))

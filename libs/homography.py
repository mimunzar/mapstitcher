from libs.util import *
import torch


def make_compute_correspondences(device, debug):
    from kornia.feature                    import LoFTR # type: ignore
    from torchvision.transforms.functional import rgb_to_grayscale
    loftr = LoFTR(pretrained = 'outdoor').to(device)
    loftr.eval()
    def compute_correspondences(image0, image1):
        with torch.inference_mode():
            matches = loftr({
                'image0': rgb_to_grayscale(image0.unsqueeze(0)).to(device),
                'image1': rgb_to_grayscale(image1.unsqueeze(0)).to(device)})
            return matches['confidence'], [matches['keypoints0'], matches['keypoints1']]
    if debug: # Plot side to side correspondences
          from matplotlib import pyplot
          def compute_correspondences_debug(image0, image1):
              conf, [kp0, kp1]  = compute_correspondences(image0, image1)
              fig , [ax0, ax1]  = pyplot.subplots(1, 2, tight_layout = True)
              line_between_axes = make_line_between_axes(fig, ax0, ax1)
              for p0, p1, c in zip(kp0.cpu(), kp1.cpu(), conf.cpu()):
                  line_between_axes(p0, p1, color = [1, 0, 0, float(c)])  # Set alpha based on confidence
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Correspondences 0'); ax0.axis('off')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Correspondences 1'); ax1.axis('off')
              pyplot.show()
              return conf, [kp0, kp1]
          return compute_correspondences_debug
    else: return compute_correspondences


def make_compute_homography(device, debug):
    from cv2 import findHomography, RANSAC
    compute_correspondences = make_compute_correspondences(device, debug)
    def compute_homography(image0, image1):
        _, [kp0, kp1]  = compute_correspondences(image0, image1)
        H, inlier_mask = findHomography(kp0.cpu().numpy(), kp1.cpu().numpy(), RANSAC, 4.0)
        return H, [kp0[1 == inlier_mask.flatten()], kp1[1 == inlier_mask.flatten()]]
    if debug:
          from matplotlib import pyplot
          def compute_homography_debug(image0, image1):
              H  , [kp0, kp1]      = compute_homography(image0, image1)
              fig, [ax0, ax1, ax2] = pyplot.subplots(1, 3, tight_layout = True)
              line_between_axes    = make_line_between_axes(fig, ax0, ax1)
              for p0, p1 in zip(kp0.cpu(), kp1.cpu()):
                  line_between_axes(p0, p1, color =[1, 0, 0])
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Inliers 0'); ax0.axis('off')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Inliers 1'); ax1.axis('off')
              ax2.imshow(warp_image(image0, H)) ; ax2.set_title('Warped')   ; ax2.axis('off')
              pyplot.show()
              return H, [kp0, kp1]
          return compute_homography_debug
    else: return compute_homography


if '__main__' == __name__:
    """Run as python -m libs.homography from a command line."""
    from pprint import pprint
    compute_homography = make_compute_homography(**{
        'device': 'cuda',
        'debug' : True}) # Expects following argparse arguments.
    read_image         = compose([
        make_read_image(),
        make_resize_image(500),
        make_square_pad  (500)])
    pprint(compute_homography(
        read_image('test_data/2610465282_cut_part_1.tif.jpg'),
        read_image('test_data/2610465282_cut_part_2.tif.jpg')))

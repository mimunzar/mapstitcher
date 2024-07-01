import torch
import cv2


def compose(funs):
    funs = list(funs)
    def compose_impl(x):
        for f in funs:
            x = f(x)
        return x
    return compose_impl


IMAGE_DTYPE = torch.float32
def make_read_image():
    """Returns a normalized image represented as [C, H, W]."""
    from PIL.Image                         import open
    from torchvision.transforms.functional import to_tensor
    def read_image(path):
        with open(path) as im:
            return to_tensor(im.convert('RGB')).to(IMAGE_DTYPE)
    return read_image


def make_resize_image(max_size):
    """Returns scaled image with a longer side of given size."""
    from torchvision.transforms.functional import resize
    from torchvision.transforms            import InterpolationMode
    bilinear = InterpolationMode.BILINEAR
    def resize_image(image):
        _, h, w = image.shape
        longer  = max(h, w)
        new_h   = round(max_size * h / longer)
        new_w   = round(max_size * w / longer)
        return resize(img           = image,
                      size          = [new_h, new_w],
                      antialias     = True,
                      interpolation = bilinear)
    return resize_image


def make_square_pad(max_size):
    """Pads mage with a black border to a square."""
    from torch import zeros
    def square_pad(image):
        c, h, w = image.shape
        y       = round((max_size - h) / 2)
        x       = round((max_size - w) / 2)
        result  = zeros([c, max_size, max_size], dtype = IMAGE_DTYPE)
        result[:, y:(y + h), x:(x + w)] = image
        return result
    return square_pad


def to_numpy_image(tensor_image):
    tensor_image[1 < tensor_image] = 1 # Silence warnings by clamping to <0, 1>
    return tensor_image.permute(1, 2, 0).numpy()


def warp_image(image, H):
    _, h, w = image.shape
    return cv2.warpPerspective(to_numpy_image(image), H, [w, h])


def make_line_between_axes(fig, src_ax, dst_ax):
    from matplotlib.patches import ConnectionPatch
    src_transform = src_ax.transData
    dst_transform = dst_ax.transData
    def line_between_axes(xy0, xy1, color):
        fig.add_artist(ConnectionPatch(
            xyA = xy0, coordsA = src_transform,
            xyB = xy1, coordsB = dst_transform, color = color))
    return line_between_axes


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
              return H, [kp0, kp1]
          return compute_homography_debug
    else: return compute_homography


if '__main__' == __name__:
    from pprint import pprint
    compute_homography = make_compute_homography('cuda', True)
    read               = compose([
        make_read_image(),
        make_resize_image(500),
        make_square_pad  (500)])

    pprint(compute_homography(
        read('digidata0/2610465282_cut_part_1.tif.jpg'),
        read('digidata0/2610465282_cut_part_2.tif.jpg')))


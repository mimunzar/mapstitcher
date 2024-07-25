def compose(funs):
    """Returns a function which is composition of given functions."""
    funs = list(funs)
    def compose_impl(x):
        for f in funs:
            x = f(x)
        return x
    return compose_impl


def make_read_image():
    """Returns a normalized image represented as [C, H, W] tensor."""
    from PIL.Image                         import open
    from torchvision.transforms.functional import to_tensor
    def read_image(path):
        with open(path) as im:
            return to_tensor(im.convert('RGB'))
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
    """Returns an square image padded with black borders."""
    from torch import zeros
    def square_pad(image):
        c, h, w = image.shape
        y       = round((max_size - h) / 2)
        x       = round((max_size - w) / 2)
        result  = zeros([c, max_size, max_size])
        result[:, y:(y + h), x:(x + w)] = image
        return result
    return square_pad


def warp_image(image, H):
    from cv2 import warpPerspective
    _, h, w = image.shape
    return warpPerspective(to_numpy_image(image), H, [w, h])


def to_numpy_image(tensor_image):
    """Converts tensor image to numpy image [C, H, W] -> [H, W, C]."""
    tensor_image[1 < tensor_image] = 1 # Silence warnings by clamping to <0, 1>
    return tensor_image.permute(1, 2, 0).numpy()



def make_line_between_axes(figure, src_axes, dst_axes):
    """Plots a line connecting two axes."""
    from matplotlib.patches import ConnectionPatch
    src_transform = src_axes.transData
    dst_transform = dst_axes.transData
    def line_between_axes(xy0, xy1, color):
        figure.add_artist(ConnectionPatch(
            xyA = xy0, coordsA = src_transform,
            xyB = xy1, coordsB = dst_transform, color = color))
    return line_between_axes
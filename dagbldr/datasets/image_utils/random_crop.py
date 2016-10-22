# Author: Kyle Kastner
# License: BSD 3-Clause
import numpy as np


def random_crop(im, random_state):
    """
    Return a single random crop
    """
    image_size = im.shape
    patch_size = (500, 400)
    patch_step = (20, 20)
    bound1 = np.arange(0, image_size[0] - patch_size[0],
                       patch_step[0]).astype("int32")
    bound2 = np.arange(patch_size[0], image_size[0],
                       patch_step[0]).astype("int32")
    g1 = list(zip(bound1, bound2))
    bound3 = np.arange(0, image_size[1] - patch_size[1],
                       patch_step[1]).astype("int32")
    bound4 = np.arange(patch_size[1], image_size[1],
                       patch_step[1]).astype("int32")
    g2 = list(zip(bound3, bound4))
    random_state.shuffle(g1)
    random_state.shuffle(g2)
    sel1 = g1[0]
    sel2 = g2[0]
    return im[sel1[0]:sel1[1], sel2[0]:sel2[1], ...]

random_state = np.random.RandomState(1999)
im = np.arange(600 * 600 * 3).reshape(600, 600, 3)
im_crop = random_crop(im, random_state)
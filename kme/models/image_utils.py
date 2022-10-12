from torch import nn


def image2d_patcher(imgs, kernel_sz, padding=0):
    imgs = nn.functional.unfold(imgs, kernel_sz, padding=padding, stride=kernel_sz)
    return imgs

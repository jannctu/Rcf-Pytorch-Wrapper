from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np

def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im

class BSDS_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS', transform=False):
        self.root = root
        self.transform = transform
        self.filelist = join(self.root, 'test.lst')
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file = self.filelist[index].rstrip()
        img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
        img = prepare_image_PIL(img)
        return img


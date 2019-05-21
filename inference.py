#!/user/bin/python
# coding=utf-8
import os, sys
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import BSDS_Loader
from model import RCF
from torch.utils.data import DataLoader, sampler
from utils import init_model
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname,exists
import scipy.io


def test_model(model, test_loader, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        #print(image.shape)
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        filename = splitext(test_list[idx])[0]
        torchvision.utils.save_image(1-results_all, join(save_dir, "%s.jpg" % filename))
        edge = {}
        edge['data'] = result
        scipy.io.savemat(join(save_dir, "%s.mat" % filename), edge)
        #result = Image.fromarray((result * 255).astype(np.uint8))
        #result.save(join(save_dir, "%s.png" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))

test_dataset = BSDS_Loader(root='../DATA/data/HED-BSDS/')
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=0, drop_last=True,shuffle=False)
with open('../DATA/data/HED-BSDS/test.lst', 'r') as f:
    test_list = f.readlines()
test_list = [split(i.rstrip())[1] for i in test_list]
assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))
print('Test size : %d' % len(test_loader))

save_dir = 'bsds'

model = RCF()
init_model(model)
#print(state_dict['conv1_1.weight'].shape)
#quit()
model.cuda()

test_model(model, test_loader, test_list, save_dir)




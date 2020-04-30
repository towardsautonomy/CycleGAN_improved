import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import cv2
import math
import time
import csv
import glob
from model import CycleGAN
from config import *
from utils import to_data

# data path
test_path = '/home/shubham/workspace/dataset/vKITTI/Scene01/clone/frames/rgb/*/*.jpg'

## create models
# call the function to get models
G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y = CycleGAN(n_res_blocks=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def changeDomain(img_x, model):
    # resize, and normalize image x
    img_x = cv2.resize(img_x, image_size)
    img_x = np.asarray((np.true_divide(img_x, 255.0) - 0.5), dtype=np.float32)
    # convert to pythorch format
    torch_img_x = np.moveaxis(img_x, -1, 0)
    # expand dim
    torch_img_x = torch_img_x[np.newaxis, ...]
    # convert to tensor
    img_x_tensor = torch.tensor(torch_img_x)

    # move image to GPU if available
    img_x_tensor = img_x_tensor.to(device)
    # set model to eval mode
    model.eval()
    # Generate image in domain B
    torch_img_y = model(img_x_tensor)
    img_y = to_data(torch_img_y[0])
    # convert back to RowxColxChannel format
    img_y = np.moveaxis(img_y, 0, 2)
    # return image
    return img_y

if __name__ == '__main__':
    # load weights
    G_XtoY.load_state_dict(torch.load(generator_x_y_weights, map_location=lambda storage, loc: storage))
    print('Loaded pretrained weights')

    # get image file names
    img_fnames = glob.glob(test_path)

    for fname in img_fnames:
        # read the image
        img_x = cv2.imread(fname)
        img_orig_size = (img_x.shape[1], img_x.shape[0])
        # convert to rgb
        img_x_rgb = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        # use the model to change domain
        img_y = changeDomain(img_x_rgb, G_XtoY)
        # rgb to bgr and resize
        img_y_bgr = cv2.cvtColor(img_y, cv2.COLOR_RGB2BGR)
        img_y_bgr = cv2.resize(img_y_bgr, img_orig_size, interpolation=cv2.INTER_LINEAR)
        # concatenate images and visualize
        img_concat = cv2.vconcat([img_x, img_y_bgr])
        cv2.imshow('Image X -> Y', img_concat)
        cv2.waitKey(0)
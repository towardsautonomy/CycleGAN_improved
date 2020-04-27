# import data loading libraries
import os
import pdb
import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

# import torch
import torch

# numpy & scipy imports
import numpy as np
import imageio
import matplotlib.pyplot as plt 
import cv2 

def checkpoint(checkpoint_dir, epoch, G_XtoY, G_YtoX, D_X, D_Y, best=False):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y."""
    if best == True:
        checkpoint_dir = os.path.join(checkpoint_dir, 'best')
    else:
        checkpoint_dir = os.path.join(checkpoint_dir, str(epoch).zfill(6))

    if not os.path.exists(checkpoint_dir):
        os.system('mkdir -p '+checkpoint_dir)
    G_XtoY_path = os.path.join(checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(checkpoint_dir, 'D_X_.pkl')
    D_Y_path = os.path.join(checkpoint_dir, 'D_Y_.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    x = ((x + 0.5)*255.0).astype(np.uint8)
    return x

def save_samples(samples_dir, epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16):
    """Saves samples from both generators X->Y and Y->X."""
    if not os.path.exists(samples_dir):
        os.system('mkdir -p '+samples_dir)
    # move input data to correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fake_Y = G_XtoY(fixed_X.to(device))
    recon_Y_X = G_YtoX(fake_Y.to(device))
    fake_X = G_YtoX(fixed_Y.to(device))
    recon_X_Y = G_XtoY(fake_X.to(device))
    
    X, fake_Y, recon_Y_X = to_data(fixed_X), to_data(fake_Y), to_data(recon_Y_X)
    Y, fake_X, recon_X_Y = to_data(fixed_Y), to_data(fake_X), to_data(recon_X_Y)

    n_rows = 4
    plt.figure(figsize=(20,16))
    # Turn off tick labels
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    for i in range(min(n_rows, batch_size)):
        plt.subplot(n_rows,2,i*2+1)
        plt.title('Original Image X   |   Translated Image    |   Reconstructed Image', fontsize=16, fontweight="bold")
        img_concat = cv2.hconcat([np.transpose(X[i,:,:,:], (1, 2, 0)),       
                                np.transpose(fake_Y[i,:,:,:], (1, 2, 0)),   
                                np.transpose(recon_Y_X[i,:,:,:], (1, 2, 0))])
        plt.imshow(img_concat)

        plt.subplot(n_rows,2,i*2+2)
        plt.title('Original Image Y   |   Translated Image    |   Reconstructed Image', fontsize=16, fontweight="bold")
        img_concat = cv2.hconcat([np.transpose(Y[i,:,:,:], (1, 2, 0)),       
                                np.transpose(fake_X[i,:,:,:], (1, 2, 0)),   
                                np.transpose(recon_X_Y[i,:,:,:], (1, 2, 0))])
        plt.imshow(img_concat)

    path = os.path.join(samples_dir, 'sample-{:06d}.png'.format(epoch))
    plt.savefig(path)
    plt.close()
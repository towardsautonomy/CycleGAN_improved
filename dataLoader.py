import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2

import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob

class GAN_DataLoader():
    def __init__(self, imageX_dir, imageY_dir, image_size=(256, 256)):
        self.image_size = image_size
        self.imageX_dir = imageX_dir
        self.imageY_dir = imageY_dir

    # normalize image
    def normalize_img(self, X):
        return (np.true_divide(X, 255.0) - 0.5)

    # denormalize image
    def denormalize_img(self, X):
        return np.asarray(np.multiply((X + 0.5), 255.0), dtype=np.uint8)

    # get list of filenames
    def _get_fnames_list(self, n_samples=-1, test_size=0.1, shuffle=True):
        fnames_x = glob.glob(self.imageX_dir)
        fnames_y = glob.glob(self.imageY_dir)

        min_n_fnames = min(len(fnames_x), len(fnames_y))
        if n_samples == -1:
            n_samples = min_n_fnames
        else:
            n_samples = min(n_samples, min_n_fnames)

        if shuffle == True:
            # shuffle the data
            p = np.random.permutation(min_n_fnames)
            fnames_x = np.asarray(fnames_x)[p]
            fnames_y = np.asarray(fnames_y)[p]

        n_test = int(float(n_samples)*test_size)
        n_train = n_samples - n_test

        fnames_x_train = fnames_x[:n_train]
        fnames_x_test = fnames_x[n_train:n_samples]
        fnames_y_train = fnames_y[:n_train]
        fnames_y_test = fnames_y[n_train:n_samples]

        # return fnames
        return fnames_x_train, fnames_y_train, fnames_x_test, fnames_y_test

    def _get_data_generator(self, fnames_x, fnames_y, batch_size=8, shuffle=True):

        assert(len(fnames_x) == len(fnames_y))
        n_samples = len(fnames_x)
        while(1):
            # shuffle the data
            if shuffle == True:
                # shuffle the data
                p = np.random.permutation(n_samples)
                fnames_x = np.asarray(fnames_x)[p]
                fnames_y = np.asarray(fnames_y)[p]

            # get a batch of data
            for offset in range(0, n_samples, batch_size):
                _X = []
                _Y = []

                fnames_x_batch = fnames_x[offset:min(n_samples,offset+batch_size)]
                fnames_y_batch = fnames_y[offset:min(n_samples,offset+batch_size)]

                for i in range(len(fnames_x_batch)):
                    # make sure a corresponding depth file exists
                    if(os.path.exists(fnames_x_batch[i]) and os.path.exists(fnames_y_batch[i])):
                        # read, resize, and normalize image x
                        img_x = cv2.cvtColor(cv2.imread(fnames_x_batch[i]), cv2.COLOR_BGR2RGB)
                        img_x = cv2.resize(img_x, self.image_size)
                        img_x = np.asarray(self.normalize_img(img_x), dtype=np.float32)

                        # convert to pythorch format
                        torch_img_x = np.moveaxis(img_x, -1, 0)

                        # read, resize, and normalize image y
                        img_y = cv2.cvtColor(cv2.imread(fnames_y_batch[i]), cv2.COLOR_BGR2RGB)
                        img_y = cv2.resize(img_y, self.image_size)
                        img_y = np.asarray(self.normalize_img(img_y), dtype=np.float32)

                        # convert to pythorch format
                        torch_img_y = np.moveaxis(img_y, -1, 0)

                        # append to the list to be returned
                        _X.append(torch_img_x)
                        _Y.append(torch_img_y)
                
                # yield
                yield torch.Tensor(np.asarray(_X)), \
                      torch.Tensor(np.asarray(_Y))

    def get_data_generator(self, n_samples=-1,  test_size=0.1, batch_size=8, shuffle=True):
        fnames_x_train, fnames_y_train, fnames_x_test, fnames_y_test = \
            self._get_fnames_list(n_samples=n_samples, test_size=test_size, shuffle=True)

        # training data generator
        dgen_train = \
            self._get_data_generator(fnames_x_train, fnames_y_train, batch_size=batch_size, shuffle=shuffle)
        # testing data generator
        dgen_test = \
            self._get_data_generator(fnames_x_test, fnames_y_test, batch_size=batch_size, shuffle=shuffle)

        return dgen_train, dgen_test

    def get_num_samples(self, n_samples=-1,  test_size=0.1):
        fnames_x_train, _, fnames_x_test, _ = \
            self._get_fnames_list(n_samples=n_samples, test_size=test_size, shuffle=False)

        return len(fnames_x_train), len(fnames_x_test)

# helper imshow function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__':
    dloader = GAN_DataLoader(imageX_dir='/home/shubham/workspace/dataset/vKITTI/Scene01/clone/frames/rgb/*/*.jpg',
                             imageY_dir='/home/shubham/workspace/dataset/KITTI/data_object_image_2/training/*/*.png')

    dloader_train, dloader_test = dloader.get_data_generator()
    dloader_train_it = iter(dloader_train)
    dloader_test_it = iter(dloader_test)

    # the "_" is a placeholder for no labels
    images_x, images_y = next(dloader_test_it)
    print(images_y.min(), images_y.max())

    # show images
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(2,1,1)
    images_x = images_x + 0.5
    imshow(torchvision.utils.make_grid(images_x, nrow=4))
    plt.title('Domain A')
    plt.subplot(2,1,2)
    images_y = images_y + 0.5
    imshow(torchvision.utils.make_grid(images_y, nrow=4))
    plt.title('Domain B')
    plt.show()
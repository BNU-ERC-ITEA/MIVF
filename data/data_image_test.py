import os
import numpy as np
import random
import torch
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms
import cv2

import utils.utils_video as utils_video
import utils.utils_for_vflare as utils_for_vflare
import utils.utils_image as util


class ImagePairedDataset(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for vflare_removal
    '''

    def __init__(self, opt):
        super(ImagePairedDataset, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.tocrop = opt['tocrop'] if opt['tocrop'] else False
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L),
                                                                                           len(self.paths_H))

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.tocrop:

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = 0
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)  # 除了255，保存成float32精度浮点数，在转换成tensor，然后切换通道为C H W

        else:
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


class ImageDataset(data.Dataset):
    '''
    # -----------------------------------------
    # Get L for vflare_removal (single_frame)
    '''

    def __init__(self, opt):
        super(ImageDataset, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.tocrop = opt['tocrop']
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'


    def __getitem__(self, index):

        # ------------------------------------
        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.tocrop:

            H, W, _ = img_L.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = 0
            patch_L = util.augment_img(patch_L, mode=mode)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L = util.uint2tensor3(patch_L)  # 除了255，保存成float32精度浮点数，在转换成tensor，然后切换通道为C H W

        else:
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L = util.uint2tensor3(img_L)

        return {'L': img_L,  'L_path': L_path}

    def __len__(self):
        return len(self.paths_L)



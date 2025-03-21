import glob
import os
import os.path
import random
import sys
sys.path.append('./')
sys.path.append('../')
import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as udata

from utils import data_augmentation
# print('i am here...')

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1, debug='N'):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]

    if debug == 'Y':
        train_dir = 'train_small'
        train_file_out = '/train_small.h5'
    else:
        train_dir = 'train'
        train_file_out = '/train.h5'
    files = glob.glob(os.path.join(data_path, train_dir, '*.png'))  # 用来匹配所有的png
    files.sort()
    h5f = h5py.File(train_file_out, 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)  # 构造不同的清晰度
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))  
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()


    print('\nprocess validation data test')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'test', '*.png'))
    files.sort()
    h5f = h5py.File('/val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:, :, 0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()

    print('training set, # samples %d\n' % train_num)
    print('testing set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True, set=''):
        super(Dataset, self).__init__()
        self.train = train
        self.set = set
        if self.train:
            filename = 'train'
            if self.set == 'debug':
                filename += '_small'
            filename += '.h5'
            print("\tset=={}, load file for train".format(filename))
            h5f = h5py.File('/' + filename, 'r')
        else:
            filename = 'val'
            filename += '.h5'
            print("\tset=={}, load file for val".format(filename))
            h5f = h5py.File('/' + filename, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            filename = 'train'
            if self.set == 'debug':
                filename += '_small'
            filename += '.h5'
            h5f = h5py.File('/' + filename, 'r')
        else:
            filename = 'val'
            filename += '.h5'
            h5f = h5py.File('/' + filename, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

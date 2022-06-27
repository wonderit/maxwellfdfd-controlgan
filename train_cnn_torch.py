import numpy as np
import pandas as pd
import os

from PIL import Image
from typing import Any, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import torch
import torch.utils.data
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
def compress_image(prev_image, n):
    if n < 2:
        return prev_image

    height = prev_image.shape[0] // n
    width = prev_image.shape[1] // n
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[n * i, n * j]

    return new_image

class CEMDataset(torch.utils.data.Dataset):
    DATASETS_TRAIN = [
        'binary_501',
        'binary_502',
        'binary_503',
        'binary_504',
        'binary_505',
        'binary_506',
        'binary_507',
        'binary_508',
        'binary_509',
        'binary_510',
        'binary_511',
        'binary_512',
        'binary_1001',
        'binary_1002',
        'binary_1003',
        'binary_rl_fix_501',
        'binary_rl_fix_502',
        'binary_rl_fix_503',
        'binary_rl_fix_504',
        'binary_rl_fix_505',
        'binary_rl_fix_506',
        'binary_rl_fix_507',
        'binary_rl_fix_508',
        'binary_rl_fix_509',
        'binary_rl_fix_510',
        'binary_rl_fix_511',
        'binary_rl_fix_512',
        'binary_rl_fix_513',
        'binary_rl_fix_514',
        'binary_rl_fix_515',
        'binary_rl_fix_516',
        'binary_rl_fix_517',
        'binary_rl_fix_518',
        'binary_rl_fix_519',
        'binary_rl_fix_520',
        'binary_rl_fix_1001',
        'binary_rl_fix_1002',
        'binary_rl_fix_1003',
        'binary_rl_fix_1004',
        'binary_rl_fix_1005',
        'binary_rl_fix_1006',
        'binary_rl_fix_1007',
        'binary_rl_fix_1008',
    ]

    DATASETS_VALID = [
        'binary_1004',
        'binary_test_1001',
        'binary_test_1002',
        'binary_rl_fix_1009',
        'binary_rl_fix_1010',
        'binary_rl_fix_1011',
        'binary_rl_fix_1012',
        'binary_rl_fix_1013',
        'binary_rl_fix_test_1001',
    ]

    DATASETS_TEST = [
        'binary_new_test_501',
        'binary_new_test_1501',
        'binary_rl_fix_1014',
        'binary_rl_fix_1015',
        'binary_rl_fix_test_1002',
        'binary_rl_fix_test_1003',
        'binary_rl_fix_test_1004',
        'binary_rl_fix_test_1005',
        'binary_test_1101',
    ]

    def __init__(self,
                 root: str,
                train: str = 'train',
                scale: int = 1,
                 ) -> None:
        self.train = train
        self.root = root
        self.scale = scale
        self.width = 200 // scale
        self.height = 100 // scale

        if self.train == 'train':
            DATAPATH = os.path.join(root, 'train')
            DATASETS = self.DATASETS_TRAIN
        elif self.train == 'valid':
            DATAPATH = os.path.join(root, 'valid')
            DATASETS = self.DATASETS_VALID
        else:
            DATAPATH = os.path.join(root, 'test')
            DATASETS = self.DATASETS_TEST

        self.data: Any = []
        self.targets = []

        print('data loading ... ')

        # load Train dataset
        for data in DATASETS:
            dataframe = pd.read_csv(os.path.join(DATAPATH, '{}.csv'.format(data)), delim_whitespace=False, header=None)
            dataset = dataframe.values

            # split into input (X) and output (Y) variables
            fileNames = dataset[:, 0]

            # 1. first try max
            dataset[:, 13:25] /= 2767.1
            self.targets.extend(dataset[:, 13:25])
            for idx, file in enumerate(fileNames):
                try:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
                    image = np.array(image).astype(np.uint8)
                except (TypeError, FileNotFoundError) as te:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))
                    try:
                        image = np.array(image).astype(np.uint8)
                    except:
                        continue
                image = compress_image(image, self.scale)
                self.data.append(np.array(image).flatten(order='C'))

        self.data = np.vstack(self.data).reshape(-1, 1, self.height, self.width)
        self.data = self.data.transpose((0, 1, 2, 3))  # convert to HWC CHW
        print(f'Data Loading Finished. len : {len(self.data)}')


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self) -> int:
        return len(self.data)

data_dir = os.path.join(os.getcwd(), 'maxwellfdfd')
cem_train = CEMDataset(data_dir, train='train', scale=5)
cem_valid = CEMDataset(data_dir, train='test', scale=5)

batch_size = 128
train_dl = DataLoader(cem_train, batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(cem_valid, batch_size, shuffle=True, pin_memory=True)

def denorm(img_tensors):
    return img_tensors * 1.

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    s = make_grid(images.detach()[:nmax], nrow=8)
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8, padding=5, pad_value=0.5).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)

import torch.nn as nn
import torch.nn.functional as F

simple_model = nn.Sequential(
    nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2, 2)
)
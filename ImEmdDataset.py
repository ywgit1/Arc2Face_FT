# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:48:29 2025

@author: User1
"""

from torch.utils import data
import os
from PIL import Image
import numpy as np
import os.path as osp
import random
from torchvision import transforms

def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    # paths = list(scandir(folder))
    # paths = [osp.join(folder, path) for path in paths]
    folder_hierarchy = list(os.walk(folder))
    paths = []
    for item in folder_hierarchy:
        _paths = [osp.join(item[0], file) for file in item[2]]
        paths = paths + _paths
    return paths

class ImEmdDataset(data.Dataset):
    def __init__(self, files, data_transform=None):
        super(ImEmdDataset, self).__init__()
        # self.files = paths_from_folder(data_path)
        # self.files = [file for file in self.files if file.lower().endswith('jpg') or file.lower().endswith('png') or file.lower().endswith('jpeg')]
        self.files = files
        if data_transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(512),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        self.transform = data_transform
        
    def __getitem__(self, index):
        im = np.array(Image.open(self.files[index]), dtype=np.float32)[:, :, :3]
        im = self.transform(im)
        name = os.path.splitext(self.files[index])[0]
        emd = np.load(name + ".npy")
        return im, emd
        
    def __len__(self):
        return len(self.files)
        
def CreateImEmdDatasets(data_path, train_transform=None, val_transform=None, split=None):
    files = paths_from_folder(data_path)
    files = [file for file in files if file.lower().endswith('jpg') or file.lower().endswith('png') or file.lower().endswith('jpeg')]
    random.seed(0)
    random.shuffle(files)
    if split is not None:
        n_train = int(len(files) * split)
    else: n_train = len(files)
    train_files = [files[i] for i in range(n_train)]
    val_files = [files[i] for i in range(n_train, len(files))]
    train_dataset = ImEmdDataset(train_files, train_transform)
    val_dataset = ImEmdDataset(val_files, val_transform)
    return {'train': train_dataset, 'val': val_dataset}
    

        
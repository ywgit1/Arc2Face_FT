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

class ImagePairDataset(data.Dataset):
    def __init__(self, frontal_path, posed_path, ids, front_map, posed_map, data_transform=None):
        super(ImagePairDataset, self).__init__()
        self.front_path = frontal_path
        self.posed_path = posed_path
        self.front_map = front_map
        self.posed_map = posed_map
        self.ids = ids
        if data_transform is None:
            data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        self.transform = data_transform
        
    def __getitem__(self, index):
        face_id = self.ids[index]
        front_file = random.choice(self.front_map[face_id])
        if len(self.posed_map[face_id]) > 0:
            # posed_file = random.choice(self.posed_map[face_id])
            sn = front_file.split('_')[-2]
            posed_file = sn + ".jpg"
            posed_full_path = os.path.join(self.posed_path, face_id, posed_file)
        else:
            sn = front_file.split('_')
            sn = '_'.join(sn[:-1])
            posed_file = sn + ".jpg"
            posed_full_path = os.path.join(self.posed_path, posed_file)
        front_im = np.array(Image.open(os.path.join(self.front_path, face_id, front_file)), dtype=np.uint8)[:, :, :3]
        front_im = self.transform(front_im)
        posed_im = Image.open(posed_full_path)
        posed_im = posed_im.resize((112, 112))
        posed_im = np.array(posed_im, dtype=np.uint8)[:, :, :3] # RGB for ArcFace ONNX model's forward()
        posed_im = self.transform(posed_im)        
        return front_im, posed_im
        
    def __len__(self):
        return len(self.ids)
        
def CreateImagePairDatasets(frontal_path, posed_path, train_transform=None, val_transform=None, split=None):
    face_ids = os.listdir(frontal_path)
    front_map = {face_id : os.listdir(os.path.join(frontal_path, face_id)) for face_id in face_ids}
    front_map = {face_id : im_list for face_id, im_list in front_map.items() if len(im_list) > 0}
    face_ids = list(front_map.keys())
    posed_map = {face_id : os.listdir(os.path.join(posed_path, face_id)) \
                 if os.path.exists(os.path.join(posed_path, face_id)) else [] for face_id in face_ids}
    random.seed(0)
    random.shuffle(face_ids)
    if split is not None:
        n_train = int(len(face_ids) * split)
    else: n_train = len(face_ids)
    train_ids = [face_ids[i] for i in range(n_train)]
    val_ids = [face_ids[i] for i in range(n_train, len(face_ids))]
    train_dataset = ImagePairDataset(frontal_path, posed_path, train_ids, front_map, posed_map, train_transform)
    val_dataset = ImagePairDataset(frontal_path, posed_path, val_ids, front_map, posed_map, val_transform)
    return {'train': train_dataset, 'val': val_dataset}
    
if __name__ == '__main__':
    ds = CreateImagePairDatasets('E:/datasets/CASIA-WebFace-arc2face-frontal', 'E:/datasets/CASIA-WebFace-arcface-114x114')
    ds = ds['train']
    batch = ds[0]

        
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:31:11 2025

@author: User1
"""
from external.facexlib.utils.face_restoration_helper import FaceRestoreHelper
import argparse
import os
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from arcface_onnx import ArcFaceONNX
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Alignment')
    parser.add_argument('-i', '--input-folder', default='E:/datasets/frontal-aligned', help='Input folder')
    parser.add_argument('-o', '--output-folder', default='E:/datasets/frontal-aligned-112x112', help='Output folder')
    parser.add_argument('-s', '--target-size', default=112, help='Target image size')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_folder, exist_ok=True)
    
    face_helper = FaceRestoreHelper(
            upscale_factor=2,
            face_size=args.target_size,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='jpg',
            use_parse=True,
            device=device,
            model_rootpath='D:/Projects/FaceSuperRes/GFPGAN/gfpgan/weights')
    
    # arcface_model = ArcFaceONNX('models/antelopev2/arcface.onnx')
        
    imgs = os.listdir(args.input_folder)
    imgs = [img for img in imgs if img.lower().endswith('jpg') or 
            img.lower().endswith('png') or img.lower().endswith('bmp') or
            img.lower().endswith('jpeg') or img.lower().endswith('ppm')]
    for img in tqdm(imgs):
        face_helper.clean_all()
        face_helper.read_image(os.path.join(args.input_folder, img))
        face_helper.get_face_landmarks_5(only_keep_largest=True, eye_dist_threshold=5)
        fn = os.path.splitext(img)[0]
        
        face_helper.align_warp_face(used_template='arcface')
        if len(face_helper.cropped_faces) != 1:
            continue
        for cropped_face in face_helper.cropped_faces:
            cv2.imwrite(os.path.join(args.output_folder, fn + '.jpg'), cropped_face)
            
        # face_helper.cropped_faces.clear()
        # face_helper.align_warp_face(used_template='arcface')
        # if len(face_helper.cropped_faces) != 1:
        #     continue
        # emd = arcface_model.get(face_helper.cropped_faces[0])
        # np.save(os.path.join(args.output_folder, fn + '.npy'), emd)
        
        
    
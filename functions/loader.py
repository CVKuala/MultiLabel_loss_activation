import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import pydicom as dicom
import cv2
import os

def load_dicom(path, size = 512):
    
    img=dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data=img.pixel_array
    data=data-np.min(data)
    if np.max(data) != 0:
        data=data/np.max(data)
    data=(data*255).astype(np.uint8)
    #data=data.astype(np.uint8)
    return cv2.cvtColor(cv2.resize(data,(32,32)), cv2.COLOR_GRAY2RGB).transpose([2,0,1]).astype(np.float32)

class dcm_data(Dataset):
    def __init__(self,image_paths,data):
        self.image_paths=image_paths
        self.data=data
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,idx):
        image_filepath=self.image_paths[idx]
        image=load_dicom(image_filepath)
        li_tensor=[]
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['patient_overall'].values))
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['C1'].values))
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['C2'].values))
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['C3'].values))
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['C4'].values))
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['C5'].values))
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['C6'].values))
        li_tensor.append(int(self.data[self.data['StudyInstanceUID']==image_filepath.split('/')[-2]]['C7'].values))
        return image,torch.Tensor(li_tensor)


def data_loaders(data_dir,data,
                     batch_size,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=False):

    

    train_dataset=dcm_data(data_dir,data)
    test_dataset=dcm_data(data_dir,data)
    
    # Create loader objects
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )
          
    return (train_loader, test_loader)

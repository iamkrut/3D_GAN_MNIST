from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from torchvision import datasets, transforms
import torch.nn as nn
import torch
import gzip

class FaceDataset( Dataset ):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__( self, csv_file, root_dir, transform=None ):
        self.frame = pd.read_csv( csv_file )
        self.root_dir = root_dir
        self.transform = transform
    def __len__( self ):
        return len( self.frame )
    def __getitem__( self, idx ):
        img_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 0 ] )
        image = Image.open( img_name )
        if self.transform:
            image = self.transform( image )
        note = self.frame.iloc[ idx, 1 ]
        return ( image, note )

class MaskFaceDataset( Dataset ):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__( self, csv_file, root_dir, transform=None ):
        self.frame = pd.read_csv( csv_file )
        self.root_dir = root_dir
        self.transform = transform
    def __len__( self ):
        return len( self.frame )
    def __getitem__( self, idx ):
        img_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 0 ] )
        image = Image.open( img_name )
        if self.transform:
            image = self.transform( image )
        mask_name = os.path.join( self.root_dir, self.frame.iloc[ idx, 1 ] )
        mask = np.load( mask_name )
        mask = mask[ 0 :: 2, 0 :: 2 ]
        return ( image, mask )

class Minst3D (Dataset):
    def __init__(self, root_dir, transform=None, batch_size=64, img_size=32):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.data_files = [os.path.join( f) for f in self.files]
        self.transform = transform
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform( image )
        voxel = np.zeros((1,self.img_size,self.img_size,self.img_size))
        voxel[:,:,:,:12] = np.stack(([image]*12), axis=3)
        return (voxel, idx)
        



    



    

import os
import numpy as np
from PIL import Image
from natsort import natsorted

def convert_one_channel(img):
    # Convert to a single channel if image has more than one channel
    if len(img.shape) > 2:
        img = img[:, :, 0]
    return img

def pre_masks(resize_shape, path):
    # Load images directly from the specified directory without extracting a zip file
    dirs = natsorted(os.listdir(path))
    
    # Initialize with the first image
    masks = Image.open(os.path.join(path, dirs[0]))
    masks = masks.resize(resize_shape, Image.LANCZOS)
    masks = convert_one_channel(np.asarray(masks))
    
    # Stack all masks together
    mask_stack = [masks]  # Start a list with the first mask
    
    for i in range(1, len(dirs)):
        img = Image.open(os.path.join(path, dirs[i]))
        img = img.resize(resize_shape, Image.LANCZOS)
        img = convert_one_channel(np.asarray(img))
        mask_stack.append(img)
    
    # Stack masks into a single numpy array
    masks = np.stack(mask_stack, axis=0)
    masks = np.reshape(masks, (len(dirs), resize_shape[0], resize_shape[1], 1))  # Keep single channel for masks
    return masks

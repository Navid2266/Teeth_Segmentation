import os
import sys
import numpy as np
from PIL import Image
from natsort import natsorted

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
default_path = script_dir + '/Original_Masks/'

def convert_one_channel(img):
    # Convert to a single channel if the image has more than one channel
    if len(img.shape) > 2:
        img = img[:, :, 0]
    return img

def pre_masks(resize_shape=(512, 512), path="C:/teeth segmentation/Main task/Data/masks"):
    dirs = natsorted(os.listdir(path))
    masks = []
    
    for filename in dirs:
        img = Image.open(os.path.join(path, filename)).convert("L")
        img = img.resize(resize_shape, Image.ANTIALIAS)
        img_array = np.asarray(img)
        img_array = convert_one_channel(img_array)
        masks.append(img_array)

    # Stack masks along a new dimension for batch processing
    masks = np.stack(masks, axis=0)
    masks = np.reshape(masks, (len(dirs), resize_shape[0], resize_shape[1], 1))  # Single channel for masks
    return masks
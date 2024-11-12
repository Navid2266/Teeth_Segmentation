import os
import numpy as np
from PIL import Image
from natsort import natsorted

def convert_one_channel(img):
    # Convert to a single channel if image has more than one channel
    if len(img.shape) > 2:
        img = img[:, :, 0]
    return img

def convert_to_rgb(img):
    # Convert single-channel (grayscale) image to 3-channel RGB by duplicating the grayscale channel
    return np.stack([img] * 3, axis=-1)

def pre_images(resize_shape, path):
    # Set path to the images directory
    path = os.path.join(path, 'Images')
    
    # Get list of all image files, excluding non-image files
    dirs = [f for f in natsorted(os.listdir(path)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    
    # Initialize arrays to hold image data and sizes
    sizes = np.zeros((len(dirs), 2))
    images_list = []
    
    for i, file_name in enumerate(dirs):
        img_path = os.path.join(path, file_name)
        img = Image.open(img_path)
        
        # Record original image size
        sizes[i, :] = img.size
        
        # Resize image and convert to one channel if necessary
        img = img.resize(resize_shape, Image.LANCZOS)
        img = convert_one_channel(np.asarray(img))
        
        # Convert grayscale image to RGB (3 channels)
        img_rgb = convert_to_rgb(img)
        
        images_list.append(img_rgb)
    
    # Stack all images into a single numpy array
    images = np.stack(images_list, axis=0)
    images = np.reshape(images, (len(dirs), resize_shape[0], resize_shape[1], 3))  # Now RGB images
    
    return images, sizes

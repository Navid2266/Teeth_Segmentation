import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Load the image
img_path = r"C:\teeth segmentation\Main task\MedSam\X-rays\train_1.png"
img_np = io.imread(img_path)

# Convert grayscale to RGB if needed
img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if len(img_np.shape) == 2 else img_np
H, W, _ = img_3c.shape

# Load YOLO annotations and convert to pixel format
yolo_annotation_path = r"C:\teeth segmentation\Main task\MedSam\Yolo\train_2.txt"
bounding_boxes = []

with open(yolo_annotation_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        _, x_center, y_center, width, height = map(float, parts)

        # Convert YOLO format to pixel coordinates
        x_center_pixel = x_center * W
        y_center_pixel = y_center * H
        width_pixel = width * W
        height_pixel = height * H

        x_min = x_center_pixel - (width_pixel / 2)
        y_min = y_center_pixel - (height_pixel / 2)
        x_max = x_center_pixel + (width_pixel / 2)
        y_max = y_center_pixel + (height_pixel / 2)

        bounding_boxes.append([x_min, y_min, x_max, y_max])

# Plot the image with bounding boxes
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(img_3c)
ax.set_title("Image with YOLO Bounding Boxes")

for box in bounding_boxes:
    x_min, y_min, x_max, y_max = box
    # Draw each bounding box
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                         edgecolor='blue', facecolor='none', lw=2)
    ax.add_patch(rect)

plt.show()

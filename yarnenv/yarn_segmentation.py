import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

def unet(input_size=(256,256,3), num_classes=1):
    inputs = Input(input_size)

    # Contracting Path
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive Path
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # Output Layer
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def load_images_and_polygon_masks(image_dir, annotation_dir, target_size=(256, 256)):
    """
    Load images and their corresponding polygon segmentation masks.

    Args:
    - image_dir (str): Directory containing the images.
    - json_dir (str): Directory containing the COCO-formatted JSON files for each image.

    Returns:
    - images (list of np.array): Loaded images.
    - masks (list of np.array): Corresponding masks for the images.
    """
    images = []
    masks = []

    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpeg')]
    
    for image_file in image_files:
        # Construct the full image file path
        img_path = os.path.join(image_dir, image_file)

        with Image.open(img_path) as img:
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(np.array(img_resized))
        
        # Find the corresponding JSON file
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(annotation_dir, json_file)
        
        # Initialize COCO api for the specific image JSON file
        coco = COCO(json_path)

         # Create a blank mask for each image
        mask = Image.new('L', target_size, 0)
        draw = ImageDraw.Draw(mask)

        # Get all annotations for the current image
        annotation_ids = coco.getAnnIds()
        annotations = coco.loadAnns(annotation_ids)

        for ann in annotations:
            for segmentation in ann['segmentation']:
                polygon = [tuple(pt) for pt in np.array(segmentation).reshape((-1, 2))]
                draw.polygon(polygon, outline=1, fill=1)
        
        masks.append(np.array(mask))
        
        # Convert lists of arrays into 4D numpy arrays
    images_array = np.stack(images, axis=0).reshape(-1, target_size[0], target_size[1], 3)
    masks_array = np.expand_dims(np.stack(masks, axis=0), -1)  # Add channel dimension to masks

    return images_array, masks_array


image_dir = '/Users/mac/Desktop/FiberFinder/yarnenv/data/training/images'
annotation_dir = '/Users/mac/Desktop/FiberFinder/yarnenv/data/training/annotations'
images, masks = load_images_and_polygon_masks(image_dir, annotation_dir)

model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(images.shape)
print(masks.shape)
model.fit(x=images, y=masks, batch_size=16, epochs=25, validation_split=0.3)

model.save('yarn_segmentation_model.h5')
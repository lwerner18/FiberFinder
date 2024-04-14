import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def load_images_and_masks(images_dir, annotations_file):
    images = []
    masks = []

    # Load COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from image filenames to their IDs
    filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}

    for img_filename in os.listdir(images_dir):
        if not img_filename.lower().endswith('.jpeg'):
            continue

        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path)
        resized_img = img.resize((256, 256))
        image_array = np.asarray(resized_img, dtype=np.float32)

         # Normalize the image
        image_array /= 255.
        images.append(image_array)

        width, height = img.size

        # Find the corresponding image ID using the filename
        img_id = filename_to_id.get(img_filename)
        if img_id is None:
            continue  # Skip if no matching ID is found

        # Filter annotations for the current image ID
        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == img_id]

        # Create a blank (black) mask image
        mask_image = Image.new('L', (width, height), 0)  # 'L' for grayscale
        draw = ImageDraw.Draw(mask_image)

        for annotation in annotations:
            for segmentation in annotation['segmentation']:
                segmentation = segmentation[0] if isinstance(segmentation[0], list) else segmentation
                polygon = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
                draw.polygon(polygon, outline=255, fill=255)

        resized_mask = mask_image.resize((256, 256))

        mask_array = np.asarray(resized_mask, dtype=np.float32)
        # Normalize the mask to have values in {0, 1}
        mask_array /= 255.
        mask_array = np.round(mask_array)  # Assuming the mask is binary, rounding values to 0 or 1
    
        # Add channel dimension to mask
        mask_array = np.expand_dims(mask_array, axis=-1)
        masks.append(mask_array)

        # Optionally, save the images and masks for visualization
        resized_img.save(f"/Users/mac/Desktop/FiberFinder/yarnenv3.11/data/view/{img_id}_image.jpeg")
        resized_mask.save(f"/Users/mac/Desktop/FiberFinder/yarnenv3.11/data/view/{img_id}_mask.jpeg")

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

# Usage
image_dir = '/Users/mac/Desktop/FiberFinder/yarnenv3.11/data/training/images'
annotations_file = '/Users/mac/Desktop/FiberFinder/yarnenv3.11/data/training/annotations.json'

input, output = load_images_and_masks(image_dir, annotations_file)

model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

data_gen_args = {
    'rotation_range': 10,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'validation_split': 0.1  # Specify the validation split here
}

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Prepare the generators for training and validation sets
train_image_generator = image_datagen.flow(input, batch_size=15, subset='training')  # For training
train_mask_generator = mask_datagen.flow(output, batch_size=15, subset='training')  # For training

val_image_generator = image_datagen.flow(input, batch_size=15, subset='validation')  # For validation
val_mask_generator = mask_datagen.flow(output, batch_size=15, subset='validation')  # For validation

# Combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

# Assuming `model` is your U-Net model
model.fit(
    train_generator,
    steps_per_epoch=len(train_image_generator),
    validation_data=val_generator,
    validation_steps=len(val_image_generator),
    epochs=25
)

model.save('yarn_segmentation_model.h5')
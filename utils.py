import os
from PIL import Image
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage1_test/'



def resize_image_to_array(image_path, size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Resizes the image to the given size and converts it to a NumPy array.
    
    :param image_path: Path to the input image.
    :param size: Desired size as a tuple (width, height).
    :return: Resized image as a NumPy array.
    """
    with Image.open(image_path) as img:
        img = img.resize(size)
        img_array = np.array(img)
        #print(image_path, img_array.shape)
        img_array = img_array[:,:,:IMG_CHANNELS]
        return img_array

def process_images_to_array(input_dir, size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Processes all images in the given directory structure, resizing them to the given size,
    and converting them to NumPy arrays.
    
    :param input_dir: Base directory containing the folders with images.
    :param size: Desired size as a tuple (width, height).
    :return: List of resized images as NumPy arrays.
    """
    train_images = []

    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            image_dir = os.path.join(root, dir_name, 'images')
            
            if os.path.isdir(image_dir):
                for image_file in os.listdir(image_dir):
                    image_path = os.path.join(image_dir, image_file)
                    
                    resized_image_array = resize_image_to_array(image_path, size)
                    train_images.append(resized_image_array)
    
    return train_images

def process_masks_to_array(input_dir, size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Processes all masks in the given directory structure, resizing them to the given size,
    and converting them to NumPy arrays.
    
    :param input_dir: Base directory containing the folders with masks.
    :param size: Desired size as a tuple (width, height).
    :return: List of resized images as NumPy arrays.
    """
    train_masks = []

    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            image_dir = os.path.join(root, dir_name, 'masks')
            
            if os.path.isdir(image_dir):
                combined_mask = []
                for image_file in os.listdir(image_dir):
                    image_path = os.path.join(image_dir, image_file)
                    with Image.open(image_path) as img:
                        img_array = np.array(img)
                        combined_mask.append(img_array)
                      
                mask = np.sum(combined_mask, axis=0)
                resized_array = np.resize(mask, size)
                train_masks.append(resized_array)
    
    return train_masks

# Example usage:
input_directory = 'data/stage1_train'
train_images = process_images_to_array(TRAIN_PATH)
train_images =  np.stack(train_images, axis=0)
train_masks = process_masks_to_array(TRAIN_PATH)
train_masks =  np.stack(train_masks, axis=0)
train_masks = np.expand_dims(train_masks, axis=-1)
test_images = process_images_to_array(TEST_PATH)
test_images = np.stack(test_images, axis=0)

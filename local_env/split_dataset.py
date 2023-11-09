import os
import shutil
import random

# Absolute paths to folders
folder_all_images = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/images/all'
folder_train_images = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/images/train'
folder_val_images = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/images/val'
folder_test_images = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/images/test'

folder_all_labels = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/labels/all'
folder_train_labels = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/labels/train'
folder_val_labels = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/labels/val'
folder_test_labels = '/Users/julesdesir/Documents/Etudes/Mines-Paris/2A/Autres/Data-challenge-Capgemini-novembre-2023/YOLO-Attempt/data/labels/test'

# Datasets sizes (total dataset size: 2350)
train_size = 1750
val_size = 250
test_size = 350

# All the dataset
all_images = [image for image in os.listdir(folder_all_images)]
all_labels = [label for label in os.listdir(folder_all_labels)]

# Training dataset
train_images = random.sample(all_images, train_size) # Random sampling

for image in train_images: # Copy in the training dataset folder
    # Images
    path_source_image = os.path.join(folder_all_images, image)
    path_destination_image = os.path.join(folder_train_images, image)
    shutil.copy(path_source_image, path_destination_image)
    print(f'Copy of {image} vers {folder_train_images}')
    
    # Labels
    label = str(image)[:-4] + '.txt'
    try:
        path_source_label = os.path.join(folder_all_labels, label)
        path_destination_label = os.path.join(folder_train_labels, label)
        shutil.copy(path_source_label, path_destination_label)
        print(f'Copy of {label} vers {folder_train_labels}')
    except FileNotFoundError: # Some images don't have a label
        pass

# Removal of the images taken for training to avoid data leakage!
all_images_without_train = []
for image in all_images:
    if image not in train_images:
        all_images_without_train.append(image)

# Validation dataset
val_images = random.sample(all_images_without_train, val_size) # Random sampling

for image in val_images: # Copy in the training dataset folder
    # Images
    path_source_image = os.path.join(folder_all_images, image)
    path_destination_image = os.path.join(folder_val_images, image)
    shutil.copy(path_source_image, path_destination_image)
    print(f'Copy of {image} vers {folder_val_images}')

    # Labels
    label = str(image)[:-4] + '.txt'
    try:
        path_source_label = os.path.join(folder_all_labels, label)
        path_destination_label = os.path.join(folder_val_labels, label)
        shutil.copy(path_source_label, path_destination_label)
        print(f'Copy of {label} vers {folder_val_labels}')
    except FileNotFoundError: # Some images don't have a label
        pass

# Removal of the images taken for training and validation to avoid data leakage!
all_images_without_train_and_val = []
for image in all_images:
    if (image not in train_images and image not in val_images):
        all_images_without_train_and_val.append(image)

# Test dataset
test_images = random.sample(all_images_without_train_and_val, test_size) # Random sampling

for image in test_images: # Copy in the test dataset folder
    # Images
    path_source_image = os.path.join(folder_all_images, image)
    path_destination_image = os.path.join(folder_test_images, image)
    shutil.copy(path_source_image, path_destination_image)
    print(f'Copy of {image} vers {folder_test_images}')

    # Labels
    label = str(image)[:-4] + '.txt'
    try:
        path_source_label = os.path.join(folder_all_labels, label)
        path_destination_label = os.path.join(folder_test_labels, label)
        shutil.copy(path_source_label, path_destination_label)
        print(f'Copy of {label} vers {folder_test_labels}')
    except FileNotFoundError: # Some images don't have a label
        pass

print("Copy ended")
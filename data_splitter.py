import os
import shutil

# Define the source directory where all the data is currently stored
source_dir = "data"  # Replace 'path_to_your_source_directory' with the actual path

# Define the destination directories for train, test, and validation sets
train_dir = "train"
test_dir = "test"
val_dir = "validation"

# Get the list of subdirectories (labels) in the source directory
labels = os.listdir(source_dir)

# Create destination directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Define the split ratios
train_split = 0.7  # 70% of the data for training
test_split = 0.2  # 20% of the data for testing
val_split = 0.1  # 10% of the data for validation


# Iterate over each label directory
for label in labels:
    label_dir = os.path.join(source_dir, label)

    # Get the list of image files for the current label
    images = os.listdir(label_dir)

    # Calculate the number of images for each split
    num_images = len(images)
    num_train = int(train_split * num_images)
    num_test = int(test_split * num_images)
    num_val = num_images - num_train - num_test

    # Split the images into train, test, and validation sets
    train_images = images[:num_train]
    test_images = images[num_train : num_train + num_test]
    val_images = images[num_train + num_test :]

    # Move images to their respective directories
    for image in train_images:
        src = os.path.join(label_dir, image)
        dst = os.path.join(train_dir, label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

    for image in test_images:
        src = os.path.join(label_dir, image)
        dst = os.path.join(test_dir, label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

    for image in val_images:
        src = os.path.join(label_dir, image)
        dst = os.path.join(val_dir, label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

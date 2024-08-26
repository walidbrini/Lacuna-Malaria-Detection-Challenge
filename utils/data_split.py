import os
import pandas as pd
import shutil

# Paths
image_dir = '/home/walid/Desktop/OpenCv/Malaria-detection/images'
test_csv_path = '/home/walid/Desktop/OpenCv/Malaria-detection/Test.csv'
train_dir = '/home/walid/Desktop/OpenCv/Malaria-detection/train'
test_dir = '/home/walid/Desktop/OpenCv/Malaria-detection/test'

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read the Test.csv file
test_df = pd.read_csv(test_csv_path)

# Get the list of test image IDs
test_image_ids = test_df['Image_ID'].tolist()

# Move images to test or train directories
for image_name in os.listdir(image_dir):
    src_path = os.path.join(image_dir, image_name)
    
    if image_name in test_image_ids:
        dest_path = os.path.join(test_dir, image_name)
    else:
        dest_path = os.path.join(train_dir, image_name)
    
    shutil.move(src_path, dest_path)

print(f"Images have been split into train and test directories.")

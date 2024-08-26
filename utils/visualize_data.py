import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import random

# Paths
image_dir = '/home/walid/Desktop/OpenCv/Malaria-detection/train'
train_csv_path = '/home/walid/Desktop/OpenCv/Malaria-detection/Train.csv'

# Read the Train.csv file
train_df = pd.read_csv(train_csv_path)

# Function to draw bounding boxes and display images
def visualize_images(df, image_dir, num_images=5):
    """
    Visualizes images with their corresponding bounding boxes.

    Parameters:
    - df (pd.DataFrame): DataFrame containing image annotations.
    - image_dir (str): Directory where images are stored.
    - num_images (int): Number of images to visualize.
    """
    # Get unique image IDs
    unique_image_ids = df['Image_ID'].unique()
    
    # Ensure num_images does not exceed the number of available images
    num_images = min(num_images, len(unique_image_ids))
    
    # Randomly select a subset of image IDs
    sample_image_ids = random.sample(list(unique_image_ids), num_images)
    
    for image_id in sample_image_ids:
        image_path = os.path.join(image_dir, image_id)
        image = cv2.imread(image_path)
        
        if image is not None:
            # Get all bounding boxes for this image
            boxes = df[df['Image_ID'] == image_id]
            
            for _, row in boxes.iterrows():
                class_label = row['class']
                confidence = row['confidence']
                ymin, xmin, ymax, xmax = int(row['ymin']), int(row['xmin']), int(row['ymax']), int(row['xmax'])
                
                # Choose color based on class
                if class_label == 'Trophozoite':
                    color = (0, 255, 0)  # Green
                elif class_label == 'WBC':
                    color = (255, 0, 0)  # Blue
                elif class_label == 'NEG':
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 0)  # Cyan for unknown classes
                
                # Draw the bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Prepare label with class and confidence
                label = f"{class_label} ({confidence})"
                
                # Increase the font scale and thickness for larger labels
                font_scale = 1.0  # Larger font size
                font_thickness = 2  # Thicker text
                
                # Calculate text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(image, (xmin, ymin - text_height - baseline), 
                                      (xmin + text_width, ymin), color, -1)
                
                # Put the label text above the bounding box
                cv2.putText(image, label, (xmin, ymin - baseline), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
            
            # Convert image from BGR (OpenCV format) to RGB (matplotlib format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display the image with bounding boxes
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            plt.title(f"Image ID: {image_id}")
            plt.axis('off')
            plt.show()
        else:
            print(f"Image {image_id} not found.")

# Visualize a sample of images with bounding boxes
visualize_images(train_df, image_dir, num_images=5)

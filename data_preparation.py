import numpy as np
import scipy.io as sio
import cv2
import os
import tensorflow as tf

class DataLoader:
    def __init__(self, gt_path, frames_path):
        """
        Initialize DataLoader with ground truth and frames paths
        
        Args:
            gt_path (str): Path to ground truth .mat file
            frames_path (str): Path to image frames
        """
        self.gt_data = sio.loadmat(gt_path)
        self.frames_path = frames_path
    
    def preprocess_image(self, img, target_size=(224, 224)):
        """
        Preprocess image for model input
        
        Args:
            img (numpy.ndarray): Input image
            target_size (tuple): Resized image dimensions
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        resized = cv2.resize(img, target_size)
        normalized = resized / 255.0
        return normalized
    
    def convert_locations_to_bboxes(self, locations, patch_size=32, img_size=(224, 224)):
        """
        Convert point locations to bounding boxes
        
        Args:
            locations (numpy.ndarray): Pedestrian locations
            patch_size (int): Size of bbox around point
            img_size (tuple): Target image dimensions
        
        Returns:
            list: Bounding boxes
        """
        bboxes = []
        for (x, y) in locations:
            bbox = [
                max(0, x - patch_size//2), 
                max(0, y - patch_size//2), 
                min(x + patch_size//2, img_size[0]), 
                min(y + patch_size//2, img_size[1])
            ]
            bboxes.append(bbox)
        return bboxes
    
    def load_dataset(self, start_idx=1, end_idx=1000, target_size=(224, 224)):
        """
        Load and preprocess dataset
        
        Args:
            start_idx (int): Starting frame index
            end_idx (int): Ending frame index
            target_size (tuple): Image resize dimensions
        
        Returns:
            tuple: Processed images and bounding boxes
        """
        X_train = []
        y_train = []
        
        for img_index in range(start_idx, end_idx + 1):
            # Load image
            img_name = os.path.join(self.frames_path, f'seq_{img_index:06d}.jpg')
            img = cv2.imread(img_name)
            
            # Extract pedestrian locations
            pedestrian_locs = self.gt_data['frame'][0][img_index-1]['loc'][0][0]
            
            # Preprocess image and annotation
            processed_img = self.preprocess_image(img, target_size)
            bboxes = self.convert_locations_to_bboxes(pedestrian_locs)
            
            X_train.append(processed_img)
            y_train.append(bboxes)
        
        return np.array(X_train), np.array(y_train)

def create_data_augmentation():
    """
    Create data augmentation pipeline
    
    Returns:
        tf.keras.Sequential: Data augmentation layers
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

# Example usage
if __name__ == "__main__":
    loader = DataLoader(
        gt_path='mall_dataset/mall_gt.mat', 
        frames_path='./mall_dataset/frames/'
    )
    X_train, y_train = loader.load_dataset()
    augmentation = create_data_augmentation()
    
    # Optional: Visualize augmented data
    X_augmented = augmentation(X_train)
    print("Dataset loaded and augmented successfully")

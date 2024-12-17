import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st

class BusPeopleCounter:
    def __init__(self, model_version='yolov8n'):
        """
        Initialize YOLO people counter
        
        Args:
            model_version (str): YOLO model version to use
        """
        # Load pre-trained YOLO model
        self.model = YOLO(f'{model_version}.pt')
        
        # Define person class index (typically 0 in COCO dataset)
        self.person_class_index = 0
    
    def count_people_in_bus(self, image):
        """
        Count people in a bus from an image
        
        Args:
            image (numpy.ndarray): Input image array
        
        Returns:
            dict: Detailed people counting results
        """
        # Detect objects using YOLO
        results = self.model(image)[0]
        
        # Filter for people only
        people_detections = [
            det for det in results.boxes.data.tolist() 
            if int(det[5]) == self.person_class_index
        ]
        
        # Count total people
        total_people = len(people_detections)
        
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        
        # Visualize detections
        for detection in people_detections:
            x1, y1, x2, y2, confidence, _ = detection
            cv2.rectangle(
                annotated_image, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 255, 0), 
                2
            )
            cv2.putText(
                annotated_image, 
                f'Person: {confidence:.2f}', 
                (int(x1), int(y1-10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
        
        # Prepare and return results
        return {
            'total_people': total_people,
            'annotated_image': annotated_image,
            'detections': people_detections,
            'confidence_scores': [det[4] for det in people_detections]
        }
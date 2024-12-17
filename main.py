import cv2
from ultralytics import YOLO
import numpy as np

class PassengerCounter:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the passenger counter with Ultralytics YOLO
        
        :param model_path: Path to YOLO model (default: YOLOv8 Nano)
        """
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Tracking variables
        self.people_entered = 0
        self.people_exited = 0
        
        # Previous frame tracking
        self.prev_frame_people = []
        
        # Counting line configuration
        self.counting_line_y = None

    def set_counting_line(self, y_position):
        """
        Set the vertical line for counting entries and exits
        
        :param y_position: Y-coordinate of the counting line
        """
        self.counting_line_y = y_position

    def count_passengers(self, frame):
        """
        Count passengers entering and exiting
        
        :param frame: Input video frame
        :return: Modified frame with counting information
        """
        # If counting line not set, use middle of the frame
        if self.counting_line_y is None:
            self.counting_line_y = frame.shape[0] // 2
        
        # Detect people using YOLO
        results = self.model(frame, stream=False)
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (frame.shape[1], self.counting_line_y), (0, 255, 0), 2)
        
        current_frame_people = []
        
        # Process detected people
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Filter for person class
                if int(box.cls[0]) == 0:  # Person class
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Calculate center y coordinate
                    center_y = (y1 + y2) // 2
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Store center y for tracking
                    current_frame_people.append(center_y)
                    
                    # Add label
                    label = f"Person: {center_y}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Count entries and exits
        for curr_person in current_frame_people:
            # Check if person wasn't in previous frame and crosses counting line
            if curr_person not in self.prev_frame_people:
                if curr_person < self.counting_line_y:
                    self.people_entered += 1
                else:
                    self.people_exited += 1
        
        # Update previous frame people
        self.prev_frame_people = current_frame_people
        
        # Display count information
        count_text = f"Entered: {self.people_entered} | Exited: {self.people_exited}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

def main():
    # Choose YOLO model (can be 'yolov8n.pt', 'yolov8s.pt', etc.)
    model_path = 'yolov8n.pt'
    
    # Open video capture (replace with your video source)
    video_path = 'bus_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Initialize passenger counter
    counter = PassengerCounter(model_path)
    
    # Optionally set custom counting line (uncomment and adjust as needed)
    counter.set_counting_line(300)  # Set line at y=300 pixels
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = counter.count_passengers(frame)
        
        # Display frame
        cv2.imshow('Passenger Counter', processed_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
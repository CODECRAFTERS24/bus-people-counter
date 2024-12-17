<<<<<<< HEAD
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
=======
import time
import cv2 
from flask import Flask, render_template, Response, request
import numpy as np
import imutils
import os


app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def find_max(k):
    d = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in d: 
            d[n] += 1
        else: 
            d[n] = 1

        # Keep track of maximum on the go
        if d[n] > maximum[1]: 
            maximum = (n,d[n])

    return maximum



UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload."""
    if 'videoFile' not in request.files:
        return "No video file provided!", 400

    file = request.files['videoFile']
    if file.filename == '':
        return "No selected file!", 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Update video file in gen function
    global video_file_path
    video_file_path = file_path
    
    return "OK", 200


# video_file_path = 'bus_video.mp4'  # Default video path

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(video_file_path)  # Use the updated video path
    avg = None
    xvalues = list()
    motion = list()
    count1 = 0
    count2 = 0



    
    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        flag = True
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
        if avg is None:
            avg = gray.copy().astype("float")
            continue
    
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 2, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if cv2.contourArea(c) < 5000:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            xvalues.append(x)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            flag = False
    	
        no_x = len(xvalues)
        
        if (no_x > 2):
            difference = xvalues[no_x - 1] - xvalues[no_x - 2]
            if(difference > 0):
                motion.append(1)
            else:
                motion.append(0)
    
        if flag is True:
            if no_x > 5:
                val, times = find_max(motion)
                if val == 1 and times >= 15:
                    count1 += 1
                else:
                    count2 += 1
                    
            xvalues = list()
            motion = list()
        
        cv2.line(frame, (260, 0), (260,480), (0,255,0), 2)
        cv2.line(frame, (420, 0), (420,480), (0,255,0), 2)	
        cv2.putText(frame, "In: {}".format(count1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Out: {}".format(count2), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Frame",frame)
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #time.sleep(0.1)
    


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

>>>>>>> master

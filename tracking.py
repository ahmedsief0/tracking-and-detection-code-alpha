from ultralytics import YOLO  # Import YOLO model
import cv2  # OpenCV for image processing

# Load YOLOv5 or YOLOv8 model (you can use yolov5s, yolov5m, yolov5l, or yolov5x)
model = YOLO('yolov8x.pt')  
# Initialize video capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(0)

# List to store trackers for each detected object
trackers = []

# Initialize frame counter
frame_counter = 0

# Load the COCO class names (you can replace with your own class names if needed)
class_names = model.names  # Class names for YOLOv5/v8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the current frame
    results = model(frame)

    # Get the bounding boxes of detected objects from YOLO
    # In YOLOv5/v8, the results are in 'results.boxes' and it contains the detections
    boxes = results[0].boxes  # Access the box attribute for results[0]

    # If the box is empty, continue to the next frame
    if len(boxes) == 0:
        continue

    # Initialize trackers only for new objects detected
    if frame_counter == 0:  # Initialize trackers on the first frame
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            tracker = cv2.TrackerCSRT_create()  # You can try KCF or MOSSE too
            tracker.init(frame, (x, y, w, h))  # Initialize tracker with the bounding box
            trackers.append(tracker)
    

    # Visualize detection results (bounding boxes from YOLO)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        class_id = int(box.cls[0])  # Get the class ID of the object
        class_name = class_names[class_id]  # Get the class name

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the class name above the bounding box
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Increment frame counter
    frame_counter += 1

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

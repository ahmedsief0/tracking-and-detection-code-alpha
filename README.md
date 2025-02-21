# Real-Time Object Detection & Tracking with YOLO and OpenCV

## 🚀 Project Overview
This project implements **real-time object detection and tracking** using **YOLOv5/YOLOv8** from Ultralytics along with OpenCV for video processing. The system detects objects in a live video stream and tracks them efficiently using OpenCV's CSRT tracker.

## 🛠️ Technologies Used
- **Python** – The primary programming language.
- **YOLOv5/YOLOv8 (Ultralytics)** – Deep learning model for object detection.
- **OpenCV** – Used for video processing and object tracking.
- **CUDA (if available)** – For accelerating deep learning inference using GPU.

## 📌 Features
- **Real-Time Object Detection** – Uses YOLO to detect objects in a video stream.
- **Object Tracking** – Employs OpenCV’s **CSRT tracker** to track detected objects across multiple frames.
- **Live Video Feed Processing** – Processes frames from a webcam or video file in real time.
- **Bounding Box Visualization** – Displays detected objects with labels on-screen.

## 🔧 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ahmedsief0/tracking-and-detection-code-alpha.git
   cd tracking-and-detection-code-alpha
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Download YOLOv5/YOLOv8 model weights:
   ```sh
   # For YOLOv5
   wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt
   
   # For YOLOv8 (optional)
   wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8x.pt
   ```

## ▶️ Usage
Run the script to start object detection and tracking:
```sh
python object_tracking.py
```

### Optional Arguments
- Change the model by modifying:
  ```python
  model = YOLO('yolov5x.pt')  # Or use 'yolov8x.pt'
  ```
- Use a video file instead of the webcam:
  ```python
  cap = cv2.VideoCapture('video.mp4')
  ```

## 📌 Future Improvements
- **Multi-Object Tracking (MOT)** – Implement ID-based tracking.
- **Edge AI Deployment** – Optimize for Raspberry Pi/Jetson Nano.
- **Custom Object Detection** – Train YOLO on a custom dataset.

## 🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests!

## 📜 License
This project is licensed under the MIT License.

## 📬 Contact
- **Author:** Ahmed Sief Eleslam  
- **LinkedIn:** [Ahmed Sief Eleslam](https://www.linkedin.com/in/ahmed-sief-eleslam-124b4a249/)  
- **GitHub:** [Your GitHub](https://github.com/ahmedsief0/)  

---
🔥 If you find this project helpful, consider giving it a ⭐ on GitHub! 🚀

# CAP6411 Project 3 (Face Detection, Tracking, and Recognition)

## Project Description
The purpose of this project was for me and my project partner (Hansi) to implement a face detection/tracker/recognition system on an embedded system [(Nvidia Jetson Xavier NX)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/) provided in the class. A [Logitech C920](https://www.logitech.com/en-gb/products/webcams/c920-pro-hd-webcam.960-001055.html) camera was used for the video feed and the system was designed such that it could detect, track, and recognize multiple faces and run at a rate of 10 Hz or more. The [YOLOv5](https://github.com/ultralytics/yolov5) [1] library was used for face detection/recognition while the [DeepSort PyTorch](https://github.com/ZQPei/deep_sort_pytorch) [2] library was used for tracking. The [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) [3] was used for training a YOLOv5S and YOLOv5L model to detect human faces. A custom dataset was then made using the trained YOLOv5L model to create bounding box labels from videos of me and Hansi in our rooms. Transfer learning was done on the YOLOv5S model so it can perform face recognition by having the model detect the classes of ('Unknown', 'Eric', 'Hansi'). A demo of the project in-action is available [here](https://drive.google.com/file/d/1W7GB4_xeZofLurtK8DYQew8FDSo4_Kti/view) and the trained YOLOv5 models and code for creating/pre-processing the datasets ('YOLO_Face', 'YOLO_Face_Recog') is available [here](https://drive.google.com/file/d/11ofw1FMFoAOMuwlvBC0nRAotmHzUrFmv/view).

## Project Structure
```
cap6411_project3
├── deep_sort_pytorch # DeepSort library used for face tracking
├── project_3         #
│   ├── flask_camera  # Used to stream C920 camera from Hansi's desktop to the embedded system
│   ├── slides        # Presentation slides from the class (planning, intermediate, final)
│   └── weights       # Trained YOLOv5S model weights to detect/recognize ('Unknown Face', 'Eric', 'Hansi')
└── yolov5            # YOLOv5 library used for face detection/recognition
```

## Project Installation
0. The project requires an Nvidia GPU with [CUDA](https://developer.nvidia.com/cuda-toolkit) support, a [Python>=3.8](https://www.python.org/) environment, and [PyTorch>=1.8](https://pytorch.org/get-started/locally/).

1. Download the project and install the required dependencies.
```
git clone https://github.com/ewatson2/cap6411_project3
cd cap6411_project3
pip install -r requirements.txt
```
2. Download the weights from the [DeepSort PyTorch](https://github.com/ZQPei/deep_sort_pytorch) [2] library.
```
mkdir -p ./deep_sort_pytorch/deep_sort/deep/checkpoint
cd ./deep_sort_pytorch/deep_sort/deep/checkpoint
# download ckpt.t7 from
# https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6
cd ../../../../
```
3. Run the face detector/tracker/recognizer using a webcam.
```
python main.py --view-img --source 0 --imgsz 640 --conf-thres 0.5 --iou-thres 0.5 --yolo_weights ./project_3/weights/yolov5s_face_recog.pt
```

## Project References
[1] [YOLOv5](https://github.com/ultralytics/yolov5)

[2] [DeepSort PyTorch](https://github.com/ZQPei/deep_sort_pytorch)

[3] [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)

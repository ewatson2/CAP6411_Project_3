# CAP6411 Project 3 (Face Detection, Tracking, and Recognition)

## Introduction


## Results


## Conclusion


## Installation
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

## References
[1] [YOLOv5 Library](https://github.com/ultralytics/yolov5)

[2] [DeepSort PyTorch](https://github.com/ZQPei/deep_sort_pytorch)

[3] [Face Recognition](https://github.com/ageitgey/face_recognition)

[4] [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)

[5] [Jetson Stats](https://github.com/rbonghi/jetson_stats)

# YOLOv9-Openvino

Implementation of YOLOv9 using Openvino Backend [YOLOv9](https://github.com/WongKinYiu/yolov9)  in C++ as well as python.

## Model

1. Download converted Model: [yolov9-c-converted](https://drive.google.com/file/d/1eBs2zlPmPoa-K2N4enTG3srXmesKQyM9/view?usp=sharing)
2. Convert your model:
``` shell
ovc yolov9-c-converted.onnx --compress_to_fp16 True --input images[1,3,640,640]
```
Note that this repository is only for model inference using openvino. Therefore, it assumes the YOLOv9 model is already trained and exported to openvino (.bin, .xml) format. 

## Setup

**Python:**
``` shell
pip install openvino
```

**CPP:**

Follow this installation guide: [guide](https://docs.openvino.ai/2023.3/openvino_docs_install_guides_installing_openvino_from_archive_windows.html)

## Usage:

**CPP:**
``` shell
yolov9-openvino-cpp.exe <model path> <data> <confidence threshold> <nms threshold>

# infer an image
yolov9-openvino-cpp.exe yolov9-c.engine test.jpg 
# infer a folder(images)
yolov9-openvino-cpp.exe yolov9-c.engine data
# infer a video
yolov9-openvino-cpp.exe yolov9-c.engine test.mp4 # the video path
```
**Python:**

Todo

### Requirement
- 2023.3.0 openvino API
- OpenCV

# YOLOv9-Openvino

Implementation of YOLOv9 using Openvino Backend [YOLOv9](https://github.com/WongKinYiu/yolov9)  in C++ as well as python.

## Model

Download [yolov9-c-converted](https://drive.google.com/file/d/1eBs2zlPmPoa-K2N4enTG3srXmesKQyM9/view?usp=sharing)

This repository is only for model inference using openvino. Therefore, it assumes the YOLOv9 model is already trained and exported to openvino (.bin, .xml) format. 

## Cpp




### Requirement
- 2023.3.0 openvino API
- OpenCV

Openvino Installation Guide : https://docs.openvino.ai/2023.3/openvino_docs_install_guides_installing_openvino_from_archive_windows.html
``` shell
pip install openvino

ovc yolov9-c-converted.onnx --compress_to_fp16 True --input images[1,3,640,640]
```

Usage:

``` shell
cd build/release

# infer an image
yolov9-tensorrt.exe yolov9-c.engine test.jpg
# infer a folder(images)
yolov9-tensorrt.exe yolov9-c.engine data
# infer a video
yolov9-tensorrt.exe yolov9-c.engine test.mp4 # the video path
```

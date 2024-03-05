#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <random>

using namespace std;
using namespace cv;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};


struct Resize
{
    cv::Mat resized_image;
    int dw;
    int dh;
};

class Yolov9
{
public:
    Yolov9(const string& model_path);
    ~Yolov9() {};

    Resize resize_and_pad(cv::Mat& img);
    void predict(cv::Mat& img, std::vector<Detection>& output);
    void draw(Mat& img, vector<Detection>& output, float dw, float dh);

    void setConf(float conf);
    void setNMS(float nms);

private:
    
    ov::CompiledModel compiled_model;

    float NMS_THRESHOLD = 0.4;
    float CONFIDENCE_THRESHOLD = 0.4;

    vector<Scalar> colors;
};

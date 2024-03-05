#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov9_openvino.h"


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

int main(int argc, char** argv)
{
    const string model_file_path{ argv[1] };
    const string path{ argv[2] };
    vector<string> imagePathList;
    bool                     isVideo{ false };
    assert(argc >= 3);

    float conf_thresh = 0.2;
    float nms_thresh = 0.3;
    if (argc > 3)
    {
        conf_thresh = std::stof(argv[3]);
    }
    else if (argc > 4)
    {
        nms_thresh = std::stof(argv[3]);
    }

    if (IsFile(path))
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    // Assume it's a folder, add logic to handle folders
    // init model
    Yolov9 model(model_file_path);
    model.setConf(conf_thresh);
    model.setNMS(nms_thresh);

    if (isVideo) {
        //path to video
        string VideoPath = path;
        // open cap
        VideoCapture cap(VideoPath);

        int width = cap.get(CAP_PROP_FRAME_WIDTH);
        int height = cap.get(CAP_PROP_FRAME_HEIGHT);

        // Create a VideoWriter object to save the processed video
        VideoWriter output_video("output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, height));
        while (1)
        {
            Mat frame;
            cap >> frame;

            if (frame.empty()) break;

            Resize res = model.resize_and_pad(frame);

            vector<Detection> bboxes;
            model.predict(res.resized_image, bboxes);
            model.draw(frame, bboxes, res.dw, res.dh);

            cv::imshow("prediction", frame);
            output_video.write(frame);
            cv::waitKey(1);
        }

        // Release resources
        cv::destroyAllWindows();
        cap.release();
        output_video.release();
    }
    else {
        // path to folder saves images
        string imageFolderPath_out = "results/";
        for (const auto& imagePath : imagePathList)
        {
            // open image
            Mat frame = imread(imagePath);
            if (frame.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            Resize res = model.resize_and_pad(frame);

            vector<Detection> bboxes;
            model.predict(res.resized_image, bboxes);
            model.draw(frame, bboxes, res.dw, res.dh);

            istringstream iss(imagePath);
            string token;
            while (getline(iss, token, '/'))
            {
            }
            imwrite(imageFolderPath_out + token, frame);
            std::cout << imageFolderPath_out + token << endl;

            cv::imshow("prediction", frame);
            cv::waitKey(0);
        }
    }

    return 0;
}

#pragma once
#include <opencv2/opencv.hpp>

class CameraReader {
public:
    CameraReader();
    ~CameraReader();

    bool open();     
    bool is_open() const;   
    bool read_frame(cv::Mat& frame); 
    void close();        

private:
    cv::VideoCapture cap_;
};

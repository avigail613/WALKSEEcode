#include "camera.h"
#include <iostream>//תקשורת בין המסך למשתמש

CameraReader::CameraReader() {}

CameraReader::~CameraReader() {
    close();
}

bool CameraReader::open() {
    cap_.open(0, cv::CAP_DSHOW); //פתיחה מהירה של המצלמה וכן מסדר רזולוציה טובה 
    if (!cap_.is_opened()) {
        std::cerr << "לא ניתן לפתוח מצלמה" << std::endl;
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  960);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    return true;
}

bool CameraReader::is_open() const {
    return cap_.is_opened();
}

bool CameraReader::read_frame(cv::Mat& frame) {
    if (!cap_.is_opened())
        return false;
    cap_ >> frame;
    return !frame.empty();
}

void CameraReader::close() {
    if (cap_.is_opened()) {
        cap_.release();
    }
}

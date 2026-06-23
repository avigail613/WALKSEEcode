#pragma once
#include <string>
#include <vector>
#include <windows.h>
#include <opencv2/opencv.hpp>

// מכשול שהגיע מ-Python (לאחר YOLO + Kalman)
struct DetectedObstacle {
    int         sector;    // 0=LEFT 1=CENTER-LEFT 2=CENTER 3=CENTER-RIGHT 4=RIGHT
    float       distance;  // מטרים
    std::string label;     // "person", "bicycle", ...
    bool        moving;    // נע או קבוע (מ-Kalman)
};

class PipeBridge {
public:
    PipeBridge();
    ~PipeBridge();

    // פותח שני Named Pipes וממתין ש-Python יתחבר
    bool open();
    bool is_open() const;
    void close();

    // שולח פריים מהמצלמה ל-Python
    // frame עובר דרך קידוד JPEG כדי להקטין נפח הנתונים
    bool send_frame(const cv::Mat& frame);

    // מקבל מ-Python את רשימת המכשולים ומחזיר אותם
    // מחזיר false אם אין תשובה עדיין
    bool receive_obstacles(std::vector<DetectedObstacle>& out);

private:
    HANDLE pipe_to_python_;    // C++ כותב, Python קורא
    HANDLE pipe_from_python_;  // Python כותב, C++ קורא
    bool   open_;

    // פרסר תשובת JSON מ-Python
    std::vector<DetectedObstacle> parse_json(const std::string& json);
};

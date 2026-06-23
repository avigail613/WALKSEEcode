#include "pipe_bridge.h"
#include <iostream>
#include <sstream>
#include <vector>

// שמות ה-Pipes — החלטיים ב-Python ו-C++ צריכים להשתמש באותם שמות
static const wchar_t* PIPE_TO_PY   = L"\\\\.\\pipe\\walksee_frame";
static const wchar_t* PIPE_FROM_PY = L"\\\\.\\pipe\\walksee_obstacles";

PipeBridge::PipeBridge() : pipe_to_python_(INVALID_HANDLE_VALUE),
                           pipe_from_python_(INVALID_HANDLE_VALUE),
                           open_(false) {}

PipeBridge::~PipeBridge() { close(); }

bool PipeBridge::open() {
    // צור Pipe ליצירה של פריימים (C++ server, Python client)
    pipe_to_python_ = CreateNamedPipeW(
        PIPE_TO_PY,
        PIPE_ACCESS_OUTBOUND,
        PIPE_TYPE_BYTE | PIPE_WAIT,
        1, 65536, 65536, 0, nullptr);

    if (pipe_to_python_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[Bridge] לא ניתן ליצור pipe_to_python" << std::endl;
        return false;
    }

    // צור Pipe לקבלת תשובות (Python server, C++ client)
    pipe_from_python_ = CreateNamedPipeW(
        PIPE_FROM_PY,
        PIPE_ACCESS_INBOUND,
        PIPE_TYPE_BYTE | PIPE_WAIT,
        1, 65536, 65536, 0, nullptr);

    if (pipe_from_python_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[Bridge] לא ניתן ליצור pipe_from_python" << std::endl;
        return false;
    }

    std::cout << "[Bridge] ממתין ש-Python יתחבר..." << std::endl;

    // מחכים ל-Python להתחבר (חוסם עד שני הצדדים מחוברים)
    ConnectNamedPipe(pipe_to_python_,   nullptr);
    ConnectNamedPipe(pipe_from_python_, nullptr);

    open_ = true;
    std::cout << "[Bridge] Python מחובר" << std::endl;
    return true;
}

bool PipeBridge::is_open() const { return open_; }

bool PipeBridge::send_frame(const cv::Mat& frame) {
    // דוחס ל-JPEG כדי להקטין את גודל הנתונים
    std::vector<uchar> jpeg_buf;
    cv::imencode(".jpg", frame, jpeg_buf);

    // שלח קודם את הגודל (4 bytes), אחר כך את הנתונים
    uint32_t size = static_cast<uint32_t>(jpeg_buf.size());
    DWORD written;
    WriteFile(pipe_to_python_, &size,          4,    &written, nullptr);
    WriteFile(pipe_to_python_, jpeg_buf.data(), size, &written, nullptr);
    return written == size;
}

bool PipeBridge::receive_obstacles(std::vector<DetectedObstacle>& out) {
    // קרא קודם את אורך ה-JSON
    uint32_t len = 0;
    DWORD read_bytes;
    if (!ReadFile(pipe_from_python_, &len, 4, &read_bytes, nullptr) || len == 0)
        return false;

    // קרא את ה-JSON
    std::string json(len, '\0');
    if (!ReadFile(pipe_from_python_, json.data(), len, &read_bytes, nullptr))
        return false;

    out = parse_json(json);
    return !out.empty();
}

// פרסר JSON ידני ללא ספרייה חיצונית
// פורמט צפוי: [{"sector":2,"distance":1.5,"label":"person","moving":false}, ...]
std::vector<DetectedObstacle> PipeBridge::parse_json(const std::string& json) {
    std::vector<DetectedObstacle> result;
    size_t pos = 0;

    while ((pos = json.find('{', pos)) != std::string::npos) {
        size_t end = json.find('}', pos);
        if (end == std::string::npos) break;
        std::string obj = json.substr(pos, end - pos + 1);

        DetectedObstacle obs{};

        // sector
        auto s = obj.find("\"sector\":");
        if (s != std::string::npos)
            obs.sector = std::stoi(obj.substr(s + 9));

        // distance
        auto d = obj.find("\"distance\":");
        if (d != std::string::npos)
            obs.distance = std::stof(obj.substr(d + 11));

        // label
        auto l = obj.find("\"label\":\"");
        if (l != std::string::npos) {
            size_t lstart = l + 9;
            size_t lend   = obj.find('"', lstart);
            obs.label = obj.substr(lstart, lend - lstart);
        }

        // moving
        obs.moving = (obj.find("\"moving\":true") != std::string::npos);

        result.push_back(obs);
        pos = end + 1;
    }
    return result;
}

void PipeBridge::close() {
    if (pipe_to_python_   != INVALID_HANDLE_VALUE) CloseHandle(pipe_to_python_);
    if (pipe_from_python_ != INVALID_HANDLE_VALUE) CloseHandle(pipe_from_python_);
    pipe_to_python_   = INVALID_HANDLE_VALUE;
    pipe_from_python_ = INVALID_HANDLE_VALUE;
    open_ = false;
}

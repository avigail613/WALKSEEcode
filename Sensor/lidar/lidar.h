#pragma once
#include <vector>//מערך שיכול לגדול אוטמטית
#include <winsock2.h>
#include <ws2tcpip.h>//לתקשורת
#pragma comment(lib, "Ws2_32.lib")

// נקודה אחת מהLiDAR
struct LidarPoint {
    float azimuth;//זוית
    float distance;//מרחק
    int   channel;//מספר לייזר
};

class LidarReader {
public:
    LidarReader();
    ~LidarReader();

    bool open();  
    bool is_open() const;
    bool read_scan(std::vector<LidarPoint>& points);//קריאת סריקה
    void close();

private:
    static constexpr const char* IP   = "192.168.1.201";
    static constexpr int PORT = 2368;
    SOCKET sock_;
    bool open_;

    void parse_packet(const uint8_t* buf, std::vector<LidarPoint>& out);
};

#pragma once
#include <vector>
#include <cstdint>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")

// נקודה אחת מהLiDAR
struct LidarPoint {
    float azimuth;   // זווית אופקית (0–359.99°)
    float distance;  // מרחק במטרים (0 = אין קריאה)
    int   channel;   // ערוץ אנכי (0–15)
};

class LidarReader {
public:
    // ip  = כתובת VLP-16 (ברירת מחדל: 192.168.1.201)
    // port = פורט UDP    (ברירת מחדל: 2368)
    explicit LidarReader(const char* ip = "192.168.1.201", int port = 2368);
    ~LidarReader();

    bool open();   // פותח socket UDP ומתחיל להאזין
    bool is_open() const;

    // קורא חבילה אחת ומחזיר את כל הנקודות שבה (~384 נקודות)
    // מחזיר false אם אין נתונים
    bool read_scan(std::vector<LidarPoint>& points);

    void close();

private:
    const char* ip_;
    int         port_;
    SOCKET      sock_;
    bool        open_;

    // פרסר חבילת VLP-16 גולמית (1206 bytes)
    void parse_packet(const uint8_t* buf, std::vector<LidarPoint>& out);
};

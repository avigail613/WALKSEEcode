#include "lidar.h"
#include <iostream>
#include <cstring>

// VLP-16: כל חבילה = 1206 bytes
// מבנה: 12 בלוקים × (2 bytes סטטוס + 32 ערוצים × 3 bytes) + timestamp + factory
static constexpr int  PACKET_SIZE   = 1206;
static constexpr int  BLOCKS_PER_PKT = 12;
static constexpr int  CHANNELS      = 16;
static constexpr float DIST_FACTOR  = 0.002f;  // יחידות VLP-16 → מטרים

// זוויות אנכיות של 16 הערוצים (מעלות)
static const float VERT_ANGLE[16] = {
    -15.0f, 1.0f, -13.0f, 3.0f, -11.0f, 5.0f, -9.0f,  7.0f,
     -7.0f, 9.0f,  -5.0f, 11.0f, -3.0f, 13.0f, -1.0f, 15.0f
};

LidarReader::LidarReader(const char* ip, int port)
    : ip_(ip), port_(port), sock_(INVALID_SOCKET), open_(false) {}

LidarReader::~LidarReader() {
    close();
    WSACleanup();
}

bool LidarReader::open() {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "[LiDAR] WSAStartup failed" << std::endl;
        return false;
    }

    sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock_ == INVALID_SOCKET) {
        std::cerr << "[LiDAR] socket failed" << std::endl;
        return false;
    }

    // timeout של 100ms — לא נחסום אם אין נתונים
    DWORD timeout = 100;
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout), sizeof(timeout));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<u_short>(port_));

    if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "[LiDAR] bind failed on port " << port_ << std::endl;
        return false;
    }

    open_ = true;
    std::cout << "[LiDAR] מאזין ל-VLP-16 על פורט " << port_ << std::endl;
    return true;
}

bool LidarReader::is_open() const {
    return open_;
}

bool LidarReader::read_scan(std::vector<LidarPoint>& points) {
    uint8_t buf[PACKET_SIZE];
    sockaddr_in sender{};
    int sender_len = sizeof(sender);

    int received = recvfrom(sock_, reinterpret_cast<char*>(buf), PACKET_SIZE, 0,
                            reinterpret_cast<sockaddr*>(&sender), &sender_len);

    if (received != PACKET_SIZE) return false;  // timeout או חבילה פגומה

    parse_packet(buf, points);
    return !points.empty();
}

void LidarReader::parse_packet(const uint8_t* buf, std::vector<LidarPoint>& out) {
    // כל בלוק מתחיל ב-offset: 0, 100, 200, ...
    for (int b = 0; b < BLOCKS_PER_PKT; ++b) {
        const uint8_t* block = buf + b * 100;

        // bytes 2-3: זווית אזימות × 100
        uint16_t az_raw = static_cast<uint16_t>(block[2]) |
                          (static_cast<uint16_t>(block[3]) << 8);
        float azimuth = az_raw / 100.0f;

        // 32 ערוצים (16 עליון + 16 תחתון בכל בלוק)
        for (int ch = 0; ch < CHANNELS; ++ch) {
            const uint8_t* ch_data = block + 4 + ch * 3;

            uint16_t dist_raw = static_cast<uint16_t>(ch_data[0]) |
                                (static_cast<uint16_t>(ch_data[1]) << 8);
            float distance = dist_raw * DIST_FACTOR;

            if (distance < 0.1f) continue;  // קריאה לא תקינה

            LidarPoint pt;
            pt.azimuth  = azimuth;
            pt.distance = distance;
            pt.channel  = ch;
            out.push_back(pt);
        }
    }
}

void LidarReader::close() {
    if (sock_ != INVALID_SOCKET) {
        closesocket(sock_);
        sock_ = INVALID_SOCKET;
        open_ = false;
    }
}

#include "lidar.h"
#include <iostream>
#include <cstring>

static constexpr int  PACKET_SIZE = 1206;//חבילה בגודל 1206 בדיוק
static constexpr int  BLOCKS_PER_PKT = 12;//12 חבילות
static constexpr int  CHANNELS= 16;//16 קרני לייזר-ערוצים
static constexpr float DIST_FACTOR = 0.002f;//מכפיל ב2MM כדי לקבל מרחק במטרים
static const float VERT_ANGLE[16] = {//16 זויות קבועות מהיצרן של הערוצים
    -15.0f, 1.0f, -13.0f, 3.0f, -11.0f, 5.0f, -9.0f,  7.0f,
     -7.0f, 9.0f,  -5.0f, 11.0f, -3.0f, 13.0f, -1.0f, 15.0f
};

LidarReader::LidarReader()
    :sock_(INVALID_SOCKET), open_(false) {}

LidarReader::~LidarReader() {
    close();
    WSACleanup();//משחרר את הזיכרון של הרשת
}

bool LidarReader::open() {
    WSADATA wsa;//פעולת רשת מאתחלת את הדרייברים
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return false;
    } 
    sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);//האזנה של UDP
    if (sock_ == INVALID_SOCKET) {
        std::cerr << "socket failed" << std::endl;
        return false;
    }
    DWORD timeout = 100;
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    sockaddr_in addr{};//פרטי הכתובת לתקשורת
    addr.sin_family= AF_INET;//כתובת מסוג IPV4
    inet_pton(AF_INET, IP, &(addr.sin_addr));
    addr.sin_port= htons(static_cast<u_short>(PORT));
    if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "bind failed on port " << PORT << std::endl;
        return false;
    }
    open_ = true;
    return true;
}

bool LidarReader::is_open() const {
    return open_;
}

bool LidarReader::read_scan(std::vector<LidarPoint>& points) {
    uint8_t buf[PACKET_SIZE];
    sockaddr_in sender{};//מבנה נתנוים לשמירת נתונים על התקשורת
    int sender_len = sizeof(sender);
    int received = recvfrom(sock_, reinterpret_cast<char*>(buf), PACKET_SIZE, 0,//כמות בייטים שהצלחנו לקרוא מהרשת
        reinterpret_cast<sockaddr*>(&sender), &sender_len);
    if (received != PACKET_SIZE)
        return false;
    char sender_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(sender.sin_addr), sender_ip, INET_ADDRSTRLEN);
    if (std::strcmp(sender_ip, IP) != 0)
        return false; 
    parse_packet(buf, points);
    return !points.empty();
}

void LidarReader::parse_packet(const uint8_t* buf, std::vector<LidarPoint>& out) {
    for (int i = 0; i < BLOCKS_PER_PKT; ++i) {
        const uint8_t* block = buf + i * 100;
        uint16_t az_raw = static_cast<uint16_t>(block[2]) |
                          (static_cast<uint16_t>(block[3]) << 8);
        float azimuth = az_raw / 100.0f;
        for (int j = 0; j < CHANNELS; ++j) {
            const uint8_t* ch_data = block + 4 + j * 3;
            uint16_t dist_raw = static_cast<uint16_t>(ch_data[0]) |
                                (static_cast<uint16_t>(ch_data[1]) << 8);
            float distance = dist_raw * DIST_FACTOR;
            if (distance < 0.1f)
                continue;  
            LidarPoint pt;
            pt.azimuth  = azimuth;
            pt.distance = distance;
            pt.channel  = j;
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

#pragma once
#include <vector>
#include <algorithm>
#include "../../Sensor/lidar/lidar.h"
#include "../../Sensor/pipe_bridge/pipe_bridge.h"

// ---- קבועים ----
constexpr int   NUM_SECTORS      = 5;      // 5 סקטורים × 72° = 360°
constexpr float SECTOR_DEG       = 72.0f;
constexpr float LIDAR_BLOCK_DIST = 2.5f;   // מטרים — ליידר מתחת ← חסום
constexpr float CAM_BLOCK_DIST   = 3.0f;   // מטרים — מצלמה מתחת ← חסום
constexpr float BLOCK_THRESHOLD  = 0.5f;   // blocked[] >= זה → סקטור חסום

// ---- מבנה הגריד ----
struct OccupancyGrid {
    float blocked[NUM_SECTORS];  // 0.0 (פנוי) עד 1.0 (חסום לחלוטין)

    OccupancyGrid() {
        for (int i = 0; i < NUM_SECTORS; i++) blocked[i] = 0.0f;
    }
};

// עזר: האם סקטור חסום?
inline bool is_blocked(const OccupancyGrid& g, int sector) {
    return g.blocked[sector] >= BLOCK_THRESHOLD;
}

// ---- בונה הגריד ----
class OccupancyGridBuilder {
public:
    // בונה גריד מנקודות LiDAR + מכשולי מצלמה (מה-PipeBridge)
    OccupancyGrid build(const std::vector<LidarPoint>&       lidar_pts,
                        const std::vector<DetectedObstacle>& obstacles);

private:
    OccupancyGrid build_lidar_grid (const std::vector<LidarPoint>&       pts);
    OccupancyGrid build_camera_grid(const std::vector<DetectedObstacle>& obs);

    // fusion: max() בכל סקטור
    static void fuse(OccupancyGrid& dst, const OccupancyGrid& src);
};

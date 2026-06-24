#include "occupancy_grid.h"
#include <cmath>

OccupancyGrid OccupancyGridBuilder::build(
    const std::vector<LidarPoint>&lidar_pts,
    const std::vector<DetectedObstacle>& obstacles)
{
    OccupancyGrid lidar_grid  = build_lidar_grid(lidar_pts);
    OccupancyGrid camera_grid = build_camera_grid(obstacles);
    fuse(lidar_grid, camera_grid);   // מיזוג: max בכל סקטור
    return lidar_grid;
}

// ==================== LiDAR → Grid ====================
// כל נקודת LiDAR: azimuth / 72 → סקטור 0-4
// blocked[s] = (נקודות < LIDAR_BLOCK_DIST) / (סה"כ נקודות בסקטור)
OccupancyGrid OccupancyGridBuilder::build_lidar_grid(
    const std::vector<LidarPoint>& pts)
{
    OccupancyGrid g;
    int total [NUM_SECTORS] = {};
    int blocked[NUM_SECTORS] = {};

    for (const auto& p : pts) {
        if (p.distance <= 0.0f) continue;

        float az = fmodf(p.azimuth, 360.0f);
        if (az < 0.0f) az += 360.0f;

        int s = static_cast<int>(az / SECTOR_DEG);
        if (s < 0)           s = 0;
        if (s >= NUM_SECTORS) s = NUM_SECTORS - 1;

        total[s]++;
        if (p.distance < LIDAR_BLOCK_DIST)
            blocked[s]++;
    }

    for (int s = 0; s < NUM_SECTORS; s++)
        if (total[s] > 0)
            g.blocked[s] = static_cast<float>(blocked[s]) / total[s];

    return g;
}

// ==================== Camera → Grid ====================
// Python כבר חישב sector (0-4) ומרחק עבור כל מכשול
// blocked[s] = 1 - distance/CAM_BLOCK_DIST  (ככל שקרוב יותר → יותר חסום)
OccupancyGrid OccupancyGridBuilder::build_camera_grid(
    const std::vector<DetectedObstacle>& obs)
{
    OccupancyGrid g;
    for (const auto& o : obs) {
        int s = o.sector;
        if (s < 0 || s >= NUM_SECTORS) continue;
        if (o.distance < CAM_BLOCK_DIST) {
            float score = 1.0f - (o.distance / CAM_BLOCK_DIST);
            if (score > g.blocked[s])
                g.blocked[s] = score;
        }
    }
    return g;
}

// Fusion 
void OccupancyGridBuilder::fuse(OccupancyGrid& dst, const OccupancyGrid& src) {
    for (int s = 0; s < NUM_SECTORS; s++)
        if (src.blocked[s] > dst.blocked[s])
            dst.blocked[s] = src.blocked[s];
}

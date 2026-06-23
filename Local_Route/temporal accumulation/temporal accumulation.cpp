#include "temporal accumulation.h"

void TemporalAccumulation::add(const OccupancyGrid& grid) {
    window_.push_back(grid);
    if (static_cast<int>(window_.size()) > TEMPORAL_WINDOW)
        window_.pop_front();  // זרוק את הפריים הישן ביותר
}

void TemporalAccumulation::clear() {
    window_.clear();
}

// ממוצע של כל הפריימים בחלון — מחליק רעש בחישתי חישי בכל סקטור
OccupancyGrid TemporalAccumulation::get_average() const {
    OccupancyGrid avg;
    if (window_.empty()) return avg;

    float sum[NUM_SECTORS] = {};

    for (const auto& g : window_)
        for (int s = 0; s < NUM_SECTORS; s++)
            sum[s] += g.blocked[s];

    float n = static_cast<float>(window_.size());
    for (int s = 0; s < NUM_SECTORS; s++)
        avg.blocked[s] = sum[s] / n;

    return avg;
}

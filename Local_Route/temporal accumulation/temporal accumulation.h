#pragma once
#include <deque>
#include "../occupancy_grid/occupancy_grid.h"

constexpr int TEMPORAL_WINDOW = 10;  // כמה פריימים לשמור בזיכרון

class TemporalAccumulation {
public:
    // מוסיף גריד חדש — אם החלון מלא, הישן נשמט
    void add(const OccupancyGrid& grid);

    // מנקה את ה-window
    void clear();

    // מחזיר ממוצע של כל הפריימים בחלון
    OccupancyGrid get_average() const;

    int size() const { return static_cast<int>(window_.size()); }

private:
    std::deque<OccupancyGrid> window_;
};

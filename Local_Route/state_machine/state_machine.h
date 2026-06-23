#pragma once
#include "../occupancy_grid/occupancy_grid.h"

// מצבי ניווט אפשריים
enum class NavState {
    STRAIGHT,     // ישרה קדימה
    TURN_LEFT,    // סטויה שמאלה
    TURN_RIGHT,   // סטויה ימינה
    STOP          // עצור — כל הכיוונים חסומים
};

// מחזיר שם קריא למצב
inline const char* nav_state_str(NavState s) {
    switch (s) {
        case NavState::STRAIGHT:   return "STRAIGHT";
        case NavState::TURN_LEFT:  return "TURN_LEFT";
        case NavState::TURN_RIGHT: return "TURN_RIGHT";
        case NavState::STOP:       return "STOP";
        default:                   return "UNKNOWN";
    }
}

class StateMachine {
public:
    // מקבל את הגריד הממוצע ומחזיר את הכיוון
    NavState update(const OccupancyGrid& grid);

    NavState current_state() const { return state_; }

private:
    NavState state_ = NavState::STRAIGHT;
};

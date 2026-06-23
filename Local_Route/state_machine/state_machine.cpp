#include "state_machine.h"

// מפת הסקטורים:
// 0=LEFT  1=CENTER-LEFT  2=CENTER  3=CENTER-RIGHT  4=RIGHT
//
// לוגיקת החלטה:
//  1. CENTER (2) פנוי                           → STRAIGHT
//  2. CENTER חסום, שמאל (0+1) פנוי           → TURN_LEFT
//  3. CENTER חסום, ימין (3+4) פנוי           → TURN_RIGHT
//  4. CENTER חסום, שניה הצדדים חסומים        → STOP
//  (CENTER-LEFT/RIGHT קל יותר מ-CENTER מוריד THRESHOLD 0.35)

NavState StateMachine::update(const OccupancyGrid& grid) {
    const float  fwd_thr  = BLOCK_THRESHOLD;   // 0.5 — חסימה לסקטור מרכזי
    const float  side_thr = 0.35f;             // סף נמוך יותר לצדדים

    bool center      = (grid.blocked[2] >= fwd_thr);
    bool center_left = (grid.blocked[1] >= fwd_thr);
    bool center_right= (grid.blocked[3] >= fwd_thr);
    bool left_clear  = (grid.blocked[0] < side_thr) && (grid.blocked[1] < side_thr);
    bool right_clear = (grid.blocked[3] < side_thr) && (grid.blocked[4] < side_thr);

    // דרך קדימה חפשיה?
    if (!center && !center_left && !center_right) {
        state_ = NavState::STRAIGHT;
        return state_;
    }

    // משהו חסום לפנינו — בחר צד פנוי
    if (left_clear && !right_clear) {
        state_ = NavState::TURN_LEFT;
    } else if (right_clear && !left_clear) {
        state_ = NavState::TURN_RIGHT;
    } else if (left_clear && right_clear) {
        // שני הצדדים פנויים — עדיף שמאל (מקובל לכיוון התנועה)
        state_ = NavState::TURN_LEFT;
    } else {
        // כל הצדדים חסומים
        state_ = NavState::STOP;
    }

    return state_;
}

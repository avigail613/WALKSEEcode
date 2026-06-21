import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import KALMAN_MAX_MISSED, KALMAN_SPEED_THR, KALMAN_MATCH_DIST

class KalmanTracker:#מחךקה-מעקב אחר עצם בודד, חישוב חיזוי
    def __init__(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self.Q = np.eye(4, dtype=np.float64) * 5.0
        self.R = np.eye(2, dtype=np.float64) * 1.0
        self.I = np.eye(4, dtype=np.float64)
        #מונים
        self.age    = 0   # כמות פריימים
        self.missed = 0   # כמה פריימים רצופים לא זיהינו את העצם

    #חיזוי
    def predict(self) -> tuple:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q#T זה שכלוף
        self.age    += 1
        self.missed += 1
        return (self.x[0], self.x[1])
    
    #עדכון 
    def update(self, cx: float, cy: float)-> None:
        z = np.array([cx, cy], dtype=np.float64) 
        S = self.H @ self.P @ self.H.T + self.R   
        K = self.P @ self.H.T @ np.linalg.inv(S)   
        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation
        self.P = (self.I - K @ self.H) @ self.P
        self.missed = 0

    #מיקום נוכחי    
    def get_predicted_position(self) -> tuple:
        return (float(self.x[0]), float(self.x[1]))

    #בדיקה אם עצם נע
    def is_moving(self) -> bool:
        speed = math.sqrt(self.x[2] ** 2 + self.x[3] ** 2)#פיתגורס
        return speed > KALMAN_SPEED_THR


class TrackerManager:#מחלקה להתאמת המכשולים השונים בין פריימים
    def __init__(self):
        self._trackers: dict = {}   
        self._next_id: int  = 0

    def update(self, obstacles: list) -> list:
        if not obstacles:
            # אין זיהויים — נבצע predict לכל tracker ונמחק ישנים
            self._predict_all()
            self._remove_lost()
            return []

        # ── שלב 1: חיזוי מיקום לכל tracker קיים ──
        predictions = self._predict_all()

        # ── שלב 2: התאמה (matching) ──
        # מחלק כל זיהוי לtracker קיים או מסמן כחדש
        matched, unmatched = self._match(predictions, obstacles)

        # ── שלב 3: עדכון trackers שנתאמו ──
        for tracker_id, obs_idx in matched.items():
            cx, cy = obstacles[obs_idx]['center']
            self._trackers[tracker_id].update(float(cx), float(cy))

        # ── שלב 4: יצירת tracker חדש לזיהויים חדשים ──
        for obs_idx in unmatched:
            cx, cy = obstacles[obs_idx]['center']
            tid = self._next_id
            self._trackers[tid] = KalmanTracker(float(cx), float(cy))
            self._next_id += 1

        # ── שלב 5: מחיקת trackers שנעלמו ──
        self._remove_lost()

        # ── שלב 6: העשרת obstacles ──
        # עדכון שדות moving ו-predicted_center לכל מכשול שנתאם
        for tracker_id, obs_idx in matched.items():
            tracker = self._trackers.get(tracker_id)
            if tracker is None:
                continue
            px, py = tracker.get_predicted_position()
            obstacles[obs_idx]['moving']            = tracker.is_moving()
            obstacles[obs_idx]['predicted_center']  = (int(px), int(py))

        return obstacles

    # ── פונקציות פנימיות ──────────────────────────────────────

    def _predict_all(self) -> dict:
        """
        מריץ predict() לכל tracker.
        מחזיר: {tracker_id: (x_חזוי, y_חזוי)}
        """
        predictions = {}
        for tid, tracker in self._trackers.items():
            predictions[tid] = tracker.predict()
        return predictions

    def _match(self, predictions: dict, obstacles: list) -> tuple:
        """
        מתאים זיהויים מ-YOLO לtrackers קיימים לפי מרחק אוקלידי.
        שיטה: Greedy — לכל זיהוי מוצא את הtracker הכי קרוב.

        מחזיר:
            matched   — {tracker_id: obs_idx}  התאמות שנמצאו
            unmatched — [obs_idx]               זיהויים ללא tracker
        """
        matched:   dict = {}
        unmatched: list = []
        used_trackers:  set = set()

        for obs_idx, obs in enumerate(obstacles):
            cx, cy = obs['center']
            best_tid  = None
            best_dist = float('inf')

            for tid, (px, py) in predictions.items():
                if tid in used_trackers:
                    continue
                dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid

            if best_tid is not None and best_dist < KALMAN_MATCH_DIST:
                matched[best_tid] = obs_idx
                used_trackers.add(best_tid)
            else:
                unmatched.append(obs_idx)

        return matched, unmatched

    def _remove_lost(self) -> None:
        """
        מוחק trackers שלא ראינו יותר מ-KALMAN_MAX_MISSED פריימים רצופים.
        """
        to_delete = [
            tid for tid, tracker in self._trackers.items()
            if tracker.missed > KALMAN_MAX_MISSED
        ]
        for tid in to_delete:
            del self._trackers[tid]

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from ultralytics import YOLO
from config import YOLO_WEIGHTS, YOLO_CONF, YOLO_OVERLAP_THR, YOLO_BLOCKED_CLASSES

# טעינת המודל פעם אחת
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO(YOLO_WEIGHTS)
    return _model


def _extract_obstacles(model, frame, sidewalk_mask, blocked_classes):
    """פונקציה פנימית — מריצה מודל אחד ומחזירה מכשולים."""
    results = model(frame, conf=YOLO_CONF, verbose=False)[0]
    obstacles = []
    if results.boxes is None:
        return obstacles

    h, w = frame.shape[:2]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        label = model.names[int(box.cls[0])]

        if label in blocked_classes:
            continue

        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(w, x2); y2c = min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        region = sidewalk_mask[y1c:y2c, x1c:x2c]
        overlap = float(region.sum()) / float(region.size) if region.size > 0 else 0.0
        print(f"[DEBUG] '{label}' {conf:.0%} overlap={overlap:.0%}")
        if overlap < YOLO_OVERLAP_THR:
            continue

        obstacles.append({
            'label':  label,
            'conf':   round(conf, 2),
            'bbox':   [x1, y1, x2, y2],
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
        })
    return obstacles


def detect_obstacles(frame, sidewalk_mask):
    """
    מזהה מכשולים בתוך המדרכה — משני מודלים במקביל.

    קלט:
        frame         — תמונה BGR (numpy [H, W, 3])
        sidewalk_mask — מסכה בינארית מ-PIDNet (numpy [H, W], 1=מדרכה)

    פלט:
        רשימת מילונים, כל אחד מכיל:
            'label'    — שם האובייקט
            'conf'     — רמת ביטחון (0-1)
            'bbox'     — [x1, y1, x2, y2] בפיקסלים
            'center'   — (cx, cy) מרכז התיבה
    """
    model = _get_model()
    return _extract_obstacles(model, frame, sidewalk_mask, YOLO_BLOCKED_CLASSES)


def draw_obstacles(frame, obstacles):
    """
    מצייר תיבות אדומות על המכשולים בתמונה.

    קלט:
        frame     — תמונה BGR מקורית
        obstacles — פלט של detect_obstacles()

    פלט:
        תמונה BGR עם תיבות מצוירות
    """
    out = frame.copy()
    for obs in obstacles:
        x1, y1, x2, y2 = obs['bbox']
        label = f"{obs['label']} {obs['conf']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out

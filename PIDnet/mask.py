import numpy as np
import torch
import cv2
from config import SIDEWALK_CLASS_ID, ROAD_CLASS_ID

# ===== צבעי ויזואליזציה לכל מחלקה (BGR) =====
CLASS_COLORS = {
    0:  (128,  64, 128),   # road       — סגול
    1:  (244,  35, 232),   # sidewalk   — ורוד
    2:  ( 70,  70,  70),   # building   — אפור כהה
    3:  (102, 102, 156),   # wall       — סגול-כחול
    4:  (190, 153, 153),   # fence      — ורוד-אפור
    5:  (153, 153, 153),   # pole       — אפור
    6:  (250, 170,  30),   # traffic light — כתום
    7:  (220, 220,   0),   # traffic sign  — צהוב
    8:  (107, 142,  35),   # vegetation — ירוק זית
    9:  (152, 251, 152),   # terrain    — ירוק בהיר
    10: ( 70, 130, 180),   # sky        — כחול שמיים
    11: (220,  20,  60),   # person     — אדום
    12: (255,   0,   0),   # rider      — אדום בהיר
    13: (  0,   0, 142),   # car        — כחול כהה
    14: (  0,   0,  70),   # truck      — כחול כהה מאוד
    15: (  0,  60, 100),   # bus        — כחול-ירוק
    16: (  0,  80, 100),   # train      — טורקיז
    17: (  0,   0, 230),   # motorcycle — כחול
    18: (119,  11,  32),   # bicycle    — בורגונדי
}


def get_segmentation_map(model_output):
    """
    ממיר את פלט הרשת למפת מחלקות.

    קלט:  tensor או numpy בצורה [1, num_classes, H, W]
    פלט:  מערך numpy [H, W] — מזהה המחלקה המנצחת לכל פיקסל
    """
    if isinstance(model_output, torch.Tensor):
        output = model_output.detach().cpu().numpy()
    else:
        output = np.array(model_output)

    # בחירת המחלקה עם הציון הגבוה ביותר לכל פיקסל
    seg_map = np.argmax(output[0], axis=0).astype(np.uint8)
    return seg_map


def get_sidewalk_mask(seg_map):
    """
    יוצר מסכה בינארית של מדרכה בלבד.

    קלט:  מפת מחלקות [H, W]
    פלט:  מערך numpy [H, W] — 1 = מדרכה, 0 = הכל השאר
    """
    binary_mask = (seg_map == SIDEWALK_CLASS_ID).astype(np.uint8)
    return binary_mask


def get_color_visualization(seg_map):
    """
    יוצר תמונה צבעונית לדיבאג — כל מחלקה בצבע שונה.

    קלט:  מפת מחלקות [H, W]
    פלט:  תמונה BGR [H, W, 3]
    """
    h, w = seg_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_img[seg_map == class_id] = color

    return color_img


def overlay_mask_on_frame(frame, seg_map, alpha=0.5):
    """
    מניח את מסכת הסגמנטציה על התמונה המקורית בשקיפות.

    קלט:
        frame   — תמונה מקורית BGR [H, W, 3]
        seg_map — מפת מחלקות [H, W]
        alpha   — שקיפות המסכה (0=שקוף, 1=אטום)
    פלט:  תמונה BGR משולבת
    """
    color_seg = get_color_visualization(seg_map)
    # שינוי גודל להתאמה לתמונה המקורית אם צריך
    if color_seg.shape[:2] != frame.shape[:2]:
        color_seg = cv2.resize(color_seg, (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(frame, 1 - alpha, color_seg, alpha, 0)
    return blended

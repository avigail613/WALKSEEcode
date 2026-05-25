import sys
import os
import torch
import cv2
import numpy as np

# ===== הוספת תיקיות ל-path =====
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)                                    # root — config.py
sys.path.insert(0, os.path.join(BASE, "PIDnet"))            # convert, mask, models
sys.path.insert(0, os.path.join(BASE, "YOLOv8-seg"))        # yolo_detect


from config import MODEL_PATH, NUM_CLASSES, SIDEWALK_CLASS_ID, CLASS_NAMES
from convert import preprocess
from mask import get_segmentation_map, get_sidewalk_mask, overlay_mask_on_frame
from models.pidnet import get_pred_model
from yolo_detect import detect_obstacles, draw_obstacles



def load_model(model_path):
    """טוען את מודל PIDNet עם המשקולות המאומנות."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] מריץ על: {device}")

    # בניית הרשת (PIDNet-S)
    model = get_pred_model(name="pidnet_s", num_classes=NUM_CLASSES)
    model.eval()

    # טעינת משקולות
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"קובץ משקולות לא נמצא: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # המשקולות יכולות להיות עטופות במפתח 'state_dict'
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("פורמט קובץ משקולות לא מוכר")

    # הצגת מפתחות לדיבאג
    sample_keys = list(state_dict.keys())[:3]
    print(f"[DEBUG] מפתחות משקולות (דוגמה): {sample_keys}")

    # הסרת קידומת 'model.' אם קיימת
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned[new_key] = v

    result = model.load_state_dict(cleaned, strict=False)
    if result.missing_keys:
        print(f"[WARN] מפתחות חסרים: {len(result.missing_keys)} — המודל אולי לא נטען מלא")
    else:
        print("[INFO] כל המשקולות נטענו בהצלחה")

    model.to(device)
    print("[INFO] מודל נטען בהצלחה")
    return model, device


def run_on_image(image_path, save_output=True):
    """
    מריץ סגמנטציה על תמונה אחת ומציג את התוצאה.
    מסמן בורוד את כל אזורי המדרכה.
    """
    # --- טעינת מודל ---
    model, device = load_model(MODEL_PATH)

    # --- טעינת תמונה מקורית ---
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"לא ניתן לטעון תמונה: {image_path}")
    print(f"[INFO] תמונה נטענה: {image_path}  ({original.shape[1]}x{original.shape[0]})")

    # --- עיבוד מקדים ---
    tensor = preprocess(original)
    tensor = tensor.to(device)

    # --- הרצת הרשת ---
    with torch.no_grad():
        output = model(tensor)

    # PIDNet מחזיר tuple באימון — בinference לוקחים את הרכיב הראשון
    if isinstance(output, (list, tuple)):
        output = output[0]

    # --- פענוח ---
    seg_map = get_segmentation_map(output)       # [H, W] מחלקה לכל פיקסל
    sidewalk_mask = get_sidewalk_mask(seg_map)   # [H, W] 1=מדרכה 0=שאר

    # --- סטטיסטיקה ---
    total_pixels    = seg_map.size
    sidewalk_pixels = int(sidewalk_mask.sum())
    sidewalk_pct    = sidewalk_pixels / total_pixels * 100
    unique_classes  = np.unique(seg_map)
    class_names_found = [f"{c}={CLASS_NAMES.get(int(c), '?')}" for c in unique_classes]
    print(f"[DEBUG] מחלקות שזוהו: {class_names_found}")
    print(f"[INFO] מדרכה זוהתה: {sidewalk_pct:.1f}% מהתמונה")

    # --- ויזואליזציה ---
    result = overlay_mask_on_frame(original, seg_map, alpha=0.5)

    # הוספת מסגרת ורודה רק על המדרכה — resize להתאמה לתמונה המקורית
    orig_h, orig_w = original.shape[:2]
    sidewalk_resized = cv2.resize(sidewalk_mask, (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)
    sidewalk_mask_3ch = np.stack([sidewalk_resized * 255,
                                   sidewalk_resized * 0,
                                   sidewalk_resized * 255], axis=-1).astype(np.uint8)
    sidewalk_overlay = cv2.addWeighted(original, 0.4, sidewalk_mask_3ch, 0.6, 0)

    # --- YOLO — זיהוי מכשולים בתוך המדרכה ---
    obstacles = detect_obstacles(original, sidewalk_resized)
    if obstacles:
        print(f"[INFO] {len(obstacles)} מכשול/ים זוהו במדרכה:")
        for obs in obstacles:
            print(f"       {obs['label']} ({obs['conf']:.0%}) — מרכז: {obs['center']}")
    else:
        print("[INFO] לא זוהו מכשולים בתוך המדרכה")

    sidewalk_overlay = draw_obstacles(sidewalk_overlay, obstacles)
    result           = draw_obstacles(result, obstacles)

    # --- טקסט על התמונה ---
    label = f"Sidewalk: {sidewalk_pct:.1f}%"
    cv2.putText(result, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # --- שמירה ---
    if save_output:
        out_seg  = image_path.replace(".", "_segmentation.")
        out_walk = image_path.replace(".", "_sidewalk_only.")
        cv2.imwrite(out_seg,  result)
        cv2.imwrite(out_walk, sidewalk_overlay)
        print(f"[INFO] תמונת סגמנטציה נשמרה: {out_seg}")
        print(f"[INFO] מסכת מדרכה נשמרה:     {out_walk}")

    # --- הצגה ---
    cv2.imshow("PIDNet — סגמנטציה מלאה", result)
    cv2.imshow("PIDNet — מדרכה בלבד",    sidewalk_overlay)
    print("[INFO] לחץ על כל מקש או סגור חלון לסגירה...")
    while True:
        key = cv2.waitKey(100)
        if key != -1:
            break
        if cv2.getWindowProperty("PIDNet — סגמנטציה מלאה", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.getWindowProperty("PIDNet — מדרכה בלבד", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("שימוש: python run.py <נתיב_לתמונה>")
        print("דוגמה: python run.py test.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    run_on_image(image_path)

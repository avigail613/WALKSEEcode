import cv2#ספריית עיבוד תמונה
import numpy as np#ספריית חישובים מתמטיים
import torch#ספרית אימון מודלים
from config import IMAGE_HEIGHT, IMAGE_WIDTH, MEAN, STD#משתנים מיובאים מהקובץ config.py


def preprocess(image_input):
    """
    ממיר תמונה גולמית מהמצלמה ל-tensor מוכן לרשת.

    קלט:  נתיב לתמונה (str)  או  מערך numpy BGR מ-OpenCV
    פלט:  tensor בצורה [1, 3, H, W]
    """
    # --- טעינה ---
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
        if frame is None:
            raise ValueError(f"לא ניתן לטעון תמונה מ: {image_input}")
    else:
        frame = image_input.copy()

    # --- אימות קלט ---
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("התמונה חייבת להיות 3 ערוצים (BGR)")
    if frame.dtype != np.uint8:
        raise ValueError("ערכי פיקסל חייבים להיות uint8 (0-255)")

    # --- שינוי גודל ---
    # הרשת מצפה לגודל קבוע שהוגדר ב-config
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT),
                       interpolation=cv2.INTER_LINEAR)

    # --- BGR → RGB ---
    # OpenCV טוען BGR, הרשת אומנה על RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- נורמליזציה: [0-255] → [0.0-1.0] → ממוצע/סטיית תקן ---
    frame = frame.astype(np.float32) / 255.0
    mean  = np.array(MEAN, dtype=np.float32)
    std   = np.array(STD,  dtype=np.float32)
    frame = (frame - mean) / std

    # --- HWC → CHW → הוספת ממד batch ---
    # הרשת מצפה לצורה [batch, channels, height, width]
    tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0)

    return tensor

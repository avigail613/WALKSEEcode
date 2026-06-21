import cv2#ספריית עיבוד תמונה
import numpy as np#ספריית חישובים מתמטיים
import torch#ספרית אימון מודלים
from config import IMAGE_HEIGHT, IMAGE_WIDTH, MEAN, STD#משתנים מיובאים מהקובץ config.py ערכי גודל וערכי נרמליזציה


def preprocess(image_input):
    #המרה של תמונה מקבל או נתיב לקובץ או פורמט של תמונה
    if isinstance(image_input, str):#אם נתיב לקובץ
        frame = cv2.imread(image_input)#פותח את הקובץ
        if frame is None:#אם קובץ לא תקין תחזיר שגיאה
            raise ValueError(f"לא ניתן לטעון תמונה מ: {image_input}")
    else:#אם קיבלת תמונה תיצור ממנו עותק 
        frame = image_input.copy()

    #אימות קלט
    if frame.ndim != 3 or frame.shape[2] != 3:#בודק שהפיקסלים הם בפורמט סטנדרטי של תמונה
        raise ValueError("התמונה חייבת להיות 3 ערוצים (BGR)")
    if frame.dtype != np.uint8:#עוד בדיקה שהפיקסלים הם בפורמט סטנדרטי של תמונה
        raise ValueError("ערכי פיקסל חייבים להיות uint8 (0-255)")

    #שינוי גודל 
    # הרשת מצפה לגודל קבוע שהוגדר ב-config האלגוריתם חותך את  התמונה בהתאם לconfig
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT),
                       interpolation=cv2.INTER_LINEAR)
     #מנרמל את התמונה שPIDNET יידע לעבוד איתה
    #cv2 מגדיר את התמונה בפורמט BGR(בסדר פיקסלים כחול ירוק אדום) אבל PIDNET מאומן על RGB(אדום ירוק כחול) השורה הזו עושה המרה כדי שPIDNET יידע לעבוד עם התמונה
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame.astype(np.float32) / 255.0#מחלק שהמספרים ינועו בים 0-1
    mean  = np.array(MEAN, dtype=np.float32)#ממוצע
    std   = np.array(STD,  dtype=np.float32)#סטיית תקן
    frame = (frame - mean) / std#מפחית ממוצע ומחלק בסטיית תקן כדי להגיע למספרים של אימון תמונה -2-2

    tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0)#משנה את הסדר מCV2 (גובה,רוחב, צבע)לPIDNET(צבע, גובה, רוחב)ומוסיף מימד כמה תמונות יש

    return tensor

def preprocess(image_input):
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
        if frame is None:
            raise ValueError(f"לא ניתן לטעון תמונה מ: {image_input}")
    else:
        frame = image_input.copy()

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("התמונה חייבת להיות 3 ערוצים (BGR)")
    if frame.dtype != np.uint8:
        raise ValueError("ערכי פיקסל חייבים להיות uint8 (0-255)")

    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT),
                       interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame.astype(np.float32) / 255.0
    mean  = np.array(MEAN, dtype=np.float32)
    std   = np.array(STD,  dtype=np.float32)
    frame = (frame - mean) / std

    tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0)

    return tensor


import os

#הגדרות PIDnet
NETIV = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(NETIV, "PIDnet", "weights", "PIDNet_S_Israel.pt")
ISRAEL_NUM_CLASSES = 4#מספר המחלקות שיש למודל(אופניים, מדרכה, כביש, רקע)

SIDEWALK_CLASS_ID = 1   # שמירת מספר המחלקה של מדרכה
ROAD_CLASS_ID = 2   # שמירת מספר המחלקה של כביש
CLASS_NAMES = {#מילון המחלקות
    0: "ofanaiim",    # אופניים (סגול בCVAT, CVAT-ID=0)
    1: "midracha",    # מדרכה  (צהוב בCVAT, CVAT-ID=1)
    2: "kvish",       # כביש   (אדום בCVAT,  CVAT-ID=2) 
    3: "acol",      
    }

#שינוי גודל התמונה לגודל קבוע שנכנס בזמן אימון המודל
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960

#ערכי נורמליזציה -מעדכן פיקסלים לצבעי האימון
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

#שימוש בGPU ולא בCPU כי מהיר יותר
USE_GPU = True

# גישה לתיקיות האימון 
DATASET_DIR = os.path.join(NETIV, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "gtFine")

#גישה למשקולות-משקולת מגיטאהב מאומנת ומשקולת ישראל שאני אימנתי
PRETRAINED_PATH = os.path.join(NETIV, "weights", "PIDNet_S_Camvid_Test.pt")
NEW_WEIGHTS_PATH = os.path.join(NETIV, "weights", "PIDNet_S_Israel.pt")

#סימון מספרים נכונים
CVAT_TO_TRAIN = {
    0: 0,   # background
    1: 0,   # acol (לא תויג בנתונים) → background
    2: 2,   # kvish       → class 2
    3: 1,   # midracha    → class 1
    4: 3,   # ofanaiim    → class 3
}

# הגדרות אימון המודל על תמונות ישראליות
TRAIN_EPOCHS = 50 #50 פעמים לעבור על כל התמונות לאימון מספיק טוב 
TRAIN_BATCH = 2 #2 מאמן כל 2 תמונות ביחד שלא יקרוס מעומס
TRAIN_LR = 1e-4 #שומרת על המשקולת המקורית שלא לאבד אותה לגמרי בזמן האימון של תמונות ישראליות אלא שינוי מזערי שלא הורס את מה שהמודל כבר יודע
TRAIN_VAL_SPLIT = 0.15 #המודל מאמן על 85% תמונות ו15% שומר לבסוף ובודק עליהם אם אימון המודל טוב או לא ומזהה מדרכה ולא תמונות מאומנות

#הגדרות YOLOv8-seg
YOLO_WEIGHTS = os.path.join(NETIV, "YOLOv8-seg", "yolov8n-seg.pt") 
YOLO_CONF = 0.1#מה הציון הנכון? מעל 0.1 בעצם נותן לכל לכל דבר 0.05 אחוז שזה תמרור 0.95 שזה אדם מה שמעל 0.1 נכון ומה שמתחת מתעלם
YOLO_OVERLAP_THR = 0.05#כמה אחוז מהמכשול שזיהה נמצא במדרכה אם גדול מ5% במדרכה זה מכשול אם קטן מתעלם בגלל שאם יהיה 0%  אז כל שניה המשתמש יצטרך לעצור על כל מכונית שיש בכביש כי yolo לא מדויק ב100%

# נתיב למפת ישראל — מורידים פעם אחת עם download_map.py ושומרים לדיסק
MAP_CACHE_PATH = os.path.join(NETIV, "map_cache", "israel_walk.graphml")

# הגדרות פילטר קלמן
KALMAN_MAX_MISSED  = 5    # כמה פריימים ממתינים לעצם לפני מחיקת הtracker שלו
KALMAN_SPEED_THR   = 3.0  # מהירות מינימלית (פיקסל/פריים) לסיווג עצם כ"נע"
KALMAN_MATCH_DIST  = 80   # מרחק מקסימלי (פיקסלים) להתאמת זיהוי לtracker קיים

# כל מה שאינו ברשימה הזו — נחשב מכשול-(רקע, תשתית קבועה, דברים שאינם חוסמים מעבר)
YOLO_BLOCKED_CLASSES = {
    # COCO
    "sky", "road", "sidewalk", "building", "wall", "floor", "ceiling",#שמיים, כביש, מדרכה,בניין, קיר, רצפה, תקרה
    "grass", "tree", "mountain", "sea", "river",#דשא, עץ, הר, ים נהר
}

#הגדרות LiDAR 
LIDAR_IP   = "192.168.1.201"
LIDAR_PORT = 2368 

# Named Pipes (C++ ↔ Python)
PIPE_FRAME     = r"\\.\pipe\walksee_frame"      # C++ כותב פריים, Python קורא
PIPE_OBSTACLES = r"\\.\pipe\walksee_obstacles"  # Python כותב מכשולים, C++ קורא

# Occupancy Grid 
NUM_SECTORS      = 5      # 5 סקטורים × 72° = 360°
SECTOR_DEG       = 72.0   # רוחב כל סקטור בדרגות
LIDAR_BLOCK_DIST = 2.5    # מטרים — ליידר מתחת לזה → חסום
CAM_BLOCK_DIST   = 3.0    # מטרים — מצלמה מתחת לזה → חסום
BLOCK_THRESHOLD  = 0.5    # blocked >= זה → הסקטור חסום

# Temporal Accumulation 
TEMPORAL_WINDOW  = 10     # כמה פריימים לשמור לממוצע

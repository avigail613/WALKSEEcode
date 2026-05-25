import os#ספריה לעבודה עם קבצים-מתקשר עם מערכת ההפעלה

######################
#  הגדרות PIDnet#
#####################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))#נתיב לתיקייה הנוכחית (היכן שנמצא הקובץ הזה)

MODEL_PATH = os.path.join(BASE_DIR, "PIDnet", "weights", "PIDNet_S_Israel.pt")#מחבר עוד תיקיות לנתיב
NUM_CLASSES = 4#מספר המחלקות שיש למודל(אופניים, מדרכה, כביש, רקע)
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
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "gtFine")

#גישה למשקולות-משקולת מגיטאהב מאומנת ומשקולת ישראל שאני אימנתי
PRETRAINED_PATH = os.path.join(BASE_DIR, "weights", "PIDNet_S_Camvid_Test.pt")
NEW_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "PIDNet_S_Israel.pt")

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

######################
#  הגדרות YOLOv8-seg#
#####################
YOLO_WEIGHTS = os.path.join(BASE_DIR, "YOLOv8-seg", "yolov8n-seg.pt") #נתיב לקובץ
YOLO_CONF = 0.1
YOLO_OVERLAP_THR = 0.05

# כל מה שאינו ברשימה הזו — נחשב מכשול
# (רקע, תשתית קבועה, דברים שאינם חוסמים מעבר)
YOLO_BLOCKED_CLASSES = {
    # COCO
    "sky", "road", "sidewalk", "building", "wall", "floor", "ceiling",
    "grass", "tree", "mountain", "sea", "river",
    # OIV7
    "Building", "Sky", "Road", "Sidewalk", "Vegetation", "Tree",
    "Land vehicle", "Tire", "Wheel",
}

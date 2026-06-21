import os#ספרייה לעבודה עם מערכת ההפעלה-ניהול קבצים ותיקיות, קריאת משתנים, הרצת פקודות מערכת ועוד
import sys#ספרייה לגישה למשתני מערכת, ניהול מודולים, יציאה מהתוכנית.מידע על גרסת Python. ועוד
import random#ספריה להגרלת  נתון-ערבוב רשימות והוצאת נתון אקראי מתוך הרשימה

import torch##ספרית אימון מודלים
import torch.nn as nn#סמודול הרשתות הנוירוניות-פריית שכבות רשת עצבית
import torch.nn.functional as F#מיועד לחישובים ברשתות נוירוניות
from torch.utils.data import DataLoader#מנהל את העבודה עבין DATASET למודל-מחלק את הנתונים, ערבב את הסדר,מאפשר לעבור על הנתונים בלולאה ועוד. כדי שהמודל לא יקרוס עם כל כך הרבה נתונים....

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))#מוסיפה את התיקייה הנוכחית לתיקיית המודלים-(__file__)הקובץ הנוכחי, os.path.abspath(__file__) מחזיר ניתוב הקובץ os.path.dirname() מחזיר רק קובץ התייקיהsys.path.insert-מוסיף לתיקיית המודלים

#טעינת הגדרות מודל מ CONFIG 
from config import (
    PRETRAINED_PATH, NEW_WEIGHTS_PATH,#נתיב למשקולות הקיימות ונתיב לשמירת המשקולות החדשות
    TRAIN_EPOCHS, TRAIN_BATCH, TRAIN_LR, TRAIN_VAL_SPLIT,#כמה פעמים לעבור על הנתונים,
    ISRAEL_NUM_CLASSES,
)
from dataset import IsraelSidewalkDataset, build_pairs
from models.pidnet import PIDNet


# טעינת backbone מהמשקולות המאומנות 
def load_pretrained_backbone(model, pretrained_path, device):
    checkpoint = torch.load(pretrained_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # הסרת קידומת 'model.' אם קיימת
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        cleaned[new_key] = v

    model_dict = model.state_dict()
    matched = {
        k: v for k, v in cleaned.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)

    total  = len(model_dict)
    loaded = len(matched)
    new    = total - loaded
    print(f"[INFO] Backbone: נטענו {loaded}/{total} שכבות | {new} שכבות head חדשות (אקראיות)")


def compute_miou(preds, labels, num_classes):
    iou_list = []
    preds_f  = preds.view(-1)
    labels_f = labels.view(-1)
    for cls in range(num_classes):
        tp    = ((preds_f == cls) & (labels_f == cls)).sum().item()
        fp    = ((preds_f == cls) & (labels_f != cls)).sum().item()
        fn    = ((preds_f != cls) & (labels_f == cls)).sum().item()
        denom = tp + fp + fn
        if denom > 0:
            iou_list.append(tp / denom)
    return sum(iou_list) / len(iou_list) if iou_list else 0.0


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] מריץ על: {device}")

    # ── Dataset ──
    all_pairs = build_pairs()
    if not all_pairs:
        raise RuntimeError("לא נמצאו זוגות תמונה/תווית. בדוק את dataset/images/ ו-dataset/gtFine/default/")

    random.seed(42)
    random.shuffle(all_pairs)
    val_n       = max(1, int(len(all_pairs) * TRAIN_VAL_SPLIT))
    train_pairs = all_pairs[val_n:]
    val_pairs   = all_pairs[:val_n]
    print(f"[INFO] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    train_ds = IsraelSidewalkDataset(train_pairs, augment=True)
    val_ds   = IsraelSidewalkDataset(val_pairs,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True,
                              num_workers=0, pin_memory=device.type == 'cuda',
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,           shuffle=False,
                              num_workers=0, pin_memory=False)

    model = PIDNet(
        m=2, n=3,
        num_classes=ISRAEL_NUM_CLASSES,
        planes=32, ppm_planes=96, head_planes=128,
        augment=True,
    )

    if os.path.exists(PRETRAINED_PATH):
        load_pretrained_backbone(model, PRETRAINED_PATH, device)
    else:
        print(f"[WARN] לא נמצאו משקולות מאומנות מראש ב-{PRETRAINED_PATH} — מתחיל מ-scratch")

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

    os.makedirs(os.path.dirname(NEW_WEIGHTS_PATH), exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            x_p, x_main, _ = model(imgs) 

            h, w   = labels.shape[-2], labels.shape[-1]
            x_main = F.interpolate(x_main, size=(h, w), mode='bilinear', align_corners=False)
            x_p    = F.interpolate(x_p,    size=(h, w), mode='bilinear', align_corners=False)

            loss = criterion(x_main, labels) + 0.4 * criterion(x_p, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        miou_vals = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                x_p, x_main, _ = model(imgs)
                h, w   = labels.shape[-2], labels.shape[-1]
                x_main = F.interpolate(x_main, size=(h, w), mode='bilinear', align_corners=False)
                preds  = x_main.argmax(dim=1)
                miou_vals.append(compute_miou(preds, labels, ISRAEL_NUM_CLASSES))

        val_miou = sum(miou_vals) / len(miou_vals) if miou_vals else 0.0
        print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS} | Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), NEW_WEIGHTS_PATH)
            print(f"  >>> נשמר מודל חדש (mIoU={best_miou:.4f}) → {NEW_WEIGHTS_PATH}")

    print(f"\n[DONE] אימון הסתיים. mIoU הטוב ביותר: {best_miou:.4f}")
    print(f"[DONE] משקולות נשמרו ב: {NEW_WEIGHTS_PATH}")


if __name__ == '__main__':
    train()

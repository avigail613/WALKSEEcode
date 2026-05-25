import os
import glob
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from config import (
    IMAGES_DIR, LABELS_DIR,
    IMAGE_HEIGHT, IMAGE_WIDTH,
    MEAN, STD, CVAT_TO_TRAIN,
)


class IsraelSidewalkDataset(Dataset):
    """
    Dataset לאימון PIDNet על תמונות ישראליות.
    קורא תמונות מ-dataset/images/ ואנוטציות מ-dataset/gtFine/default/.
    """

    def __init__(self, pairs, augment=False):
        """
        pairs   : list of (image_path, label_path)
        augment : האם להפעיל augmentation אקראי
        """
        self.pairs   = pairs
        self.augment = augment

        # בניית טבלת remap (גודל 256 — מכסה את כל ערכי uint8)
        self._remap = torch.zeros(256, dtype=torch.long)
        for cvat_id, train_id in CVAT_TO_TRAIN.items():
            self._remap[cvat_id] = train_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        # ── טעינה ──
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)   # grayscale / palette

        # ── שינוי גודל ──
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
        lbl = lbl.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)

        # ── Augmentation ──
        if self.augment:
            # היפוך אופקי אקראי
            if random.random() > 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            # שינוי בהירות/ניגודיות אקראי
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
                img = TF.adjust_contrast(img,   random.uniform(0.75, 1.25))

        # ── תמונה → tensor מנורמל ──
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, MEAN, STD)

        # ── תווית → remap → tensor ──
        lbl_arr = torch.from_numpy(np.array(lbl, dtype=np.uint8)).long()
        lbl_arr = self._remap[lbl_arr]

        return img_t, lbl_arr


def build_pairs(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR):
    """
    מחפש זוגות תואמים:
        images/X.jpg  ←→  gtFine/default/X_gtFine_labelIds.png
    מחזיר list of (image_path, label_path).
    """
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    image_files += glob.glob(os.path.join(images_dir, '*.jpeg'))
    image_files += glob.glob(os.path.join(images_dir, '*.png'))
    image_files += glob.glob(os.path.join(images_dir, '*.webp'))

    pairs = []
    missing = 0
    for img_path in sorted(image_files):
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, f"{stem}_gtFine_labelIds.png")
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
        else:
            missing += 1

    if missing:
        print(f"[WARN] {missing} תמונות ללא קובץ תווית — מדולגות")
    print(f"[INFO] {len(pairs)} זוגות תמונה-תווית נמצאו")
    return pairs


import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import time
from pathlib import Path

# =========================
# 基本設定
# =========================

BASE_DIR = Path.home() / "workspace" / "Final"

MODEL_PATH = BASE_DIR / "target_person_classifier.pth"

SAVE_ROOT = BASE_DIR / "auto_labeled_crops"
TARGET_DIR = SAVE_ROOT / "auto_target_person"
OTHER_DIR = SAVE_ROOT / "auto_other_people"
UNCERTAIN_DIR = SAVE_ROOT / "uncertain"

TARGET_DIR.mkdir(parents=True, exist_ok=True)
OTHER_DIR.mkdir(parents=True, exist_ok=True)
UNCERTAIN_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 參數
# =========================

CAMERA_SOURCE = 0

IMG_SIZE = 224

YOLO_CONF = 0.5
CLASSIFIER_CONF_TH = 0.85

SAVE_INTERVAL_SECONDS = 5
FRAME_DIFF_THRESHOLD = 3.0

MAX_PER_CLASS = 500
MAX_FRAMES = 100000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)
print("Save root:", SAVE_ROOT)

# =========================
# 載入模型
# =========================

yolo_model = YOLO("yolov8n.pt")

classifier = models.mobilenet_v3_small(weights=None)
in_features = classifier.classifier[3].in_features
classifier.classifier[3] = nn.Linear(in_features, 2)

classifier.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

classifier = classifier.to(DEVICE)
classifier.eval()

CLASS_NAMES = ["other_people", "target_person"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# 工具函式
# =========================

def count_images(folder):
    return len([
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.softmax(outputs, dim=1)

        pred = outputs.argmax(dim=1).item()
        confidence = probs[0][pred].item()

    return CLASS_NAMES[pred], confidence


def choose_save_dir(label, confidence):
    if confidence < CLASSIFIER_CONF_TH:
        return UNCERTAIN_DIR, "uncertain"

    if label == "target_person":
        return TARGET_DIR, "target_person"

    return OTHER_DIR, "other_people"


def save_crop(crop, save_dir, prefix, frame_idx, person_idx, confidence):
    if count_images(save_dir) >= MAX_PER_CLASS:
        return False

    timestamp = int(time.time() * 1000)

    filename = (
        f"{prefix}_frame{frame_idx}_person{person_idx}_"
        f"conf{confidence:.2f}_{timestamp}.jpg"
    )

    save_path = save_dir / filename
    cv2.imwrite(str(save_path), crop)

    return True


def get_small_gray(frame):
    small = cv2.resize(frame, (160, 120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def frame_difference_score(gray1, gray2):
    diff = cv2.absdiff(gray1, gray2)
    score = diff.mean()
    return score

# =========================
# 開啟攝影機
# =========================

cap = cv2.VideoCapture(CAMERA_SOURCE)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

frame_idx = 0
saved_total = 0
last_save_time = 0
last_saved_gray = None

print("Camera started.")
print("SSH headless mode: no cv2.imshow()")
print(f"Save interval: {SAVE_INTERVAL_SECONDS} seconds")
print(f"Frame diff threshold: {FRAME_DIFF_THRESHOLD}")
print(f"Program will stop after {MAX_FRAMES} frames.")

# =========================
# 主迴圈
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame.")
        break

    frame_idx += 1
    current_time = time.time()

    should_check = current_time - last_save_time >= SAVE_INTERVAL_SECONDS

    if should_check:

        current_gray = get_small_gray(frame)

        if last_saved_gray is None:
            diff_score = 999.0
            changed = True
        else:
            diff_score = frame_difference_score(current_gray, last_saved_gray)
            changed = diff_score >= FRAME_DIFF_THRESHOLD

        if changed:

            last_save_time = current_time
            last_saved_gray = current_gray.copy()

            results = yolo_model(
                frame,
                classes=[0],
                conf=YOLO_CONF,
                verbose=False
            )

            person_idx = 0

            for r in results:

                if r.boxes is None:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()

                for box in boxes:

                    person_idx += 1

                    x1, y1, x2, y2 = box.astype(int)

                    h, w = frame.shape[:2]

                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    label, confidence = classify_crop(crop)

                    save_dir, prefix = choose_save_dir(label, confidence)

                    saved = save_crop(
                        crop=crop,
                        save_dir=save_dir,
                        prefix=prefix,
                        frame_idx=frame_idx,
                        person_idx=person_idx,
                        confidence=confidence
                    )

                    if saved:
                        saved_total += 1

            print(
                f"[SAVE] Frame:{frame_idx} | "
                f"diff:{diff_score:.2f} | "
                f"saved total:{saved_total} | "
                f"target:{count_images(TARGET_DIR)} | "
                f"other:{count_images(OTHER_DIR)} | "
                f"uncertain:{count_images(UNCERTAIN_DIR)}"
            )

        else:
            last_save_time = current_time

            print(
                f"[SKIP] Frame:{frame_idx} | "
                f"diff:{diff_score:.2f} < {FRAME_DIFF_THRESHOLD}"
            )

    if frame_idx % 300 == 0:
        print(
            f"Frame:{frame_idx} | "
            f"saved total:{saved_total} | "
            f"target:{count_images(TARGET_DIR)} | "
            f"other:{count_images(OTHER_DIR)} | "
            f"uncertain:{count_images(UNCERTAIN_DIR)}"
        )

    if frame_idx >= MAX_FRAMES:
        print("Reached MAX_FRAMES. Stop.")
        break

cap.release()

print("Finished.")
print("Saved root:", SAVE_ROOT)
print("Target:", count_images(TARGET_DIR))
print("Other:", count_images(OTHER_DIR))
print("Uncertain:", count_images(UNCERTAIN_DIR))

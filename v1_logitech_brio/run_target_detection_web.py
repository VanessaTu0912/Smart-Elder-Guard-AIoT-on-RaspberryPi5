# -*- coding: utf-8 -*-

from flask import Flask, Response
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque
import threading
import time
import json

from awscrt import mqtt
from awsiot import mqtt_connection_builder

app = Flask(__name__)

AWS_ENDPOINT = "a1w6f81xx8o24k-ats.iot.us-east-1.amazonaws.com"
AWS_CLIENT_ID = "rpi5-door-detector"

AWS_CERT_PATH = "certs/device.pem.crt"
AWS_KEY_PATH = "certs/private.pem.key"
AWS_ROOT_CA_PATH = "certs/AmazonRootCA1.pem"

AWS_TOPIC = "home/door/exit"

mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=AWS_ENDPOINT,
    cert_filepath=AWS_CERT_PATH,
    pri_key_filepath=AWS_KEY_PATH,
    ca_filepath=AWS_ROOT_CA_PATH,
    client_id=AWS_CLIENT_ID,
    clean_session=False,
    keep_alive_secs=30
)

print("Connecting to AWS IoT...")
mqtt_connection.connect().result()
print("Connected to AWS IoT")


def publish_exit_event(track_id):
    payload = {
        "event": "target_exit",
        "track_id": int(track_id),
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    mqtt_connection.publish(
        topic=AWS_TOPIC,
        payload=json.dumps(payload),
        qos=mqtt.QoS.AT_LEAST_ONCE
    )

    print("Published to AWS IoT:", payload)


yolo_model = YOLO("yolov8n.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

classifier = models.mobilenet_v3_small(weights=None)
in_features = classifier.classifier[3].in_features
classifier.classifier[3] = nn.Linear(in_features, 2)

classifier.load_state_dict(
    torch.load(
        "target_person_classifier.pth",
        map_location=DEVICE
    )
)

classifier = classifier.to(DEVICE)
classifier.eval()

CLASS_NAMES = ["other_people", "target_person"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CAMERA_SOURCE = 0

#INSIDE_BOX = (190, 120, 330, 250)
INSIDE_BOX = (185, 200, 620, 460)
DOOR_BOX = (60, 50, 185, 160)

TARGET_CONF_TH = 0.65
ID_HISTORY_LEN = 10
ID_TARGET_CONFIRM = 4

YOLO_CONF = 0.6
OVERLAP_IOU_TH = 0.15

MIN_BOX_AREA = 3000
MIN_ASPECT_RATIO = 0.75

DISAPPEAR_TRIGGER_FRAMES = 8

SHOW_ALL_PEOPLE = True

id_votes = defaultdict(lambda: deque(maxlen=ID_HISTORY_LEN))
dynamic_target_ids = set()
confirmed_target_ids = set()

track_state = defaultdict(lambda: "unknown")
last_seen_frame = {}
door_enter_frame = {}
already_exit_notified = set()

frame_index = 0

latest_frame = None
frame_lock = threading.Lock()

active_clients = 0


def open_camera():
    for cam_id in [CAMERA_SOURCE, 0, 1, 2, 3]:
        cap = cv2.VideoCapture(cam_id)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if cap.isOpened():
            print("Camera opened:", cam_id)
            return cap

        cap.release()

    print("Cannot open any camera")
    return None


def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def upper_body_point(person_box):
    x1, y1, x2, y2 = person_box
    px = int((x1 + x2) / 2)
    py = int(y1 + (y2 - y1) * 0.25)
    return px, py


def calc_iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter_area / (areaA + areaB - inter_area + 1e-6)


def has_overlap(current_box, all_boxes, overlap_th=0.15):
    for other_box in all_boxes:
        if np.array_equal(current_box, other_box):
            continue

        if calc_iou(current_box, other_box) > overlap_th:
            return True

    return False


def is_valid_person_box(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1

    area = w * h

    if area < MIN_BOX_AREA:
        return False

    aspect_ratio = h / max(w, 1)

    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    return True


def classify_person(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.softmax(outputs, dim=1)

        pred = outputs.argmax(dim=1).item()
        confidence = probs[0][pred].item()

    return CLASS_NAMES[pred], confidence


def update_target_identity(track_id, raw_label, confidence):
    is_target_now = (
        raw_label == "target_person" and
        confidence >= TARGET_CONF_TH
    )

    id_votes[track_id].append(1 if is_target_now else 0)
    vote_count = sum(id_votes[track_id])

    if vote_count >= ID_TARGET_CONFIRM:
        dynamic_target_ids.add(track_id)
        confirmed_target_ids.add(track_id)
        return "target_person", vote_count

    if track_id in confirmed_target_ids:
        dynamic_target_ids.add(track_id)
        return "target_person", vote_count

    dynamic_target_ids.discard(track_id)
    return "uncertain", vote_count

def draw_box(frame, box, text, color):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def camera_worker():
    global frame_index, latest_frame

    cap = open_camera()

    if cap is None:
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            time.sleep(0.1)
            continue

        frame_index += 1

        results = yolo_model.track(
            source=frame,
            classes=[0],
            conf=YOLO_CONF,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        if len(results) == 0:
            continue

        r = results[0]
        frame = r.orig_img.copy()
        frame_h, frame_w = frame.shape[:2]

        draw_box(frame, INSIDE_BOX, "INSIDE ROI", (255, 255, 0))
        draw_box(frame, DOOR_BOX, "DOOR ROI", (255, 0, 255))

        exit_detected_this_frame = False

        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            int_boxes = boxes.astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)

                x1 = max(0, min(x1, frame_w - 1))
                x2 = max(0, min(x2, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                y2 = max(0, min(y2, frame_h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                if not is_valid_person_box(x1, y1, x2, y2):
                    continue

                current_box = np.array([x1, y1, x2, y2])

                if has_overlap(current_box, int_boxes, OVERLAP_IOU_TH):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        frame,
                        f"ID {track_id} overlap_skip",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2
                    )
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                raw_label, confidence = classify_person(crop)

                label, vote_count = update_target_identity(
                    track_id,
                    raw_label,
                    confidence
                )

                ux, uy = upper_body_point((x1, y1, x2, y2))
                cv2.circle(frame, (ux, uy), 8, (0, 0, 255), -1)

                if label == "target_person":
                    last_seen_frame[track_id] = frame_index

                    in_inside = point_in_box(ux, uy, INSIDE_BOX)
                    in_door = point_in_box(ux, uy, DOOR_BOX)

                    if in_inside:
                        track_state[track_id] = "inside"

                    elif in_door and track_state[track_id] == "inside":
                        track_state[track_id] = "door"
                        door_enter_frame[track_id] = frame_index

                if label == "target_person":
                    color = (0, 255, 0)
                    display_label = "TARGET_PERSON"
                elif raw_label == "target_person":
                    color = (0, 255, 255)
                    display_label = "target_candidate"
                else:
                    color = (180, 180, 180)
                    display_label = "other_or_uncertain"

                if SHOW_ALL_PEOPLE or label == "target_person":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        frame,
                        f"Track ID {track_id} | {display_label}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"conf:{confidence:.2f} vote:{vote_count}/{ID_HISTORY_LEN}",
                        (x1, min(frame_h - 35, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"state:{track_state[track_id]}",
                        (x1, min(frame_h - 10, y2 + 45)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )

        for track_id in list(confirmed_target_ids):
            if track_id in already_exit_notified:
                continue

            if track_state[track_id] == "door":
                last_seen = last_seen_frame.get(track_id, frame_index)
                disappear_frames = frame_index - last_seen

                if disappear_frames >= DISAPPEAR_TRIGGER_FRAMES:
                    print(f"TARGET EXITED: ID {track_id}")

                    try:
                        publish_exit_event(track_id)
                    except Exception as e:
                        print("AWS publish failed:", e)

                    already_exit_notified.add(track_id)
                    track_state[track_id] = "outside"
                    exit_detected_this_frame = True

        if exit_detected_this_frame:
            cv2.putText(
                frame,
                "TARGET EXITED!",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        cv2.putText(
            frame,
            f"Dynamic target IDs: {list(dynamic_target_ids)}",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Track ID is temporary, not identity ID",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {frame_index}",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)

        if ret:
            with frame_lock:
                latest_frame = buffer.tobytes()


def generate_stream():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )

        time.sleep(0.03)


@app.route("/")
def index():
    return """
    <html>
    <body>
    <h2>Raspberry Pi Target Person Detection Stream</h2>
    <img src="/video">
    </body>
    </html>
    """


@app.route("/video")
def video():
    global active_clients

    active_clients += 1
    print("Viewer connected:", active_clients)

    def stream():
        global active_clients

        try:
            yield from generate_stream()
        finally:
            active_clients -= 1
            print("Viewer disconnected:", active_clients)

    return Response(
        stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    t = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    t.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
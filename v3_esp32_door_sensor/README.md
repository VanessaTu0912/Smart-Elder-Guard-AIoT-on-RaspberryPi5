# Version 3 – ESP32 Door Sensor Integration

## Overview

Version 3 introduces an ESP32-based magnetic door sensor to improve the reliability of exit-event detection.

Unlike traditional sensor-triggered systems, the camera remains continuously active and performs real-time monitoring. The ESP32 door sensor serves as an auxiliary verification mechanism to confirm whether the entrance door was actually opened.

By combining computer vision and door-sensor information, the system can reduce false alarms caused by temporary tracking loss, occlusion, or misclassification.

---

## System Architecture

```text
Magnetic Door Sensor
          ↓
     ESP32 DevKit V1
          ↓
 Wi-Fi / HTTP Communication
          ↓
      Raspberry Pi 5
          ↓
 Door Event Verification
          ↓
  Exit Detection Logic
          ↓
     AWS IoT Core
          ↓
      AWS Lambda
          ↓
 LINE Messaging API
          ↓
 Caregiver Notification
```

The ESP32 transmits door-open and door-closed events to Raspberry Pi through HTTP requests.

The camera continuously performs:

- Person Detection
- Person Tracking
- Target Person Classification

The door sensor is only used as an additional verification signal.

---

## Hardware Components

| Component | Description |
|------------|------------|
| ESP32 DevKit V1 | Reads door sensor status and sends events through Wi-Fi |
| Magnetic Door Sensor | Detects door open/close status |
| Raspberry Pi 5 | Receives door events and integrates them into exit detection logic |
| Wi-Fi Router | Provides communication between ESP32 and Raspberry Pi |

---

## Communication Flow

```text
Door Opens
     ↓
ESP32 Detects Event
     ↓
HTTP POST Request
     ↓
Raspberry Pi Flask Server
     ↓
Update Door Status
     ↓
Assist Exit Verification
```

---

## HTTP API

### Endpoint

```text
POST /door
```

### Example Payload

Door Open:

```json
{
  "door": "open"
}
```

Door Closed:

```json
{
  "door": "closed"
}
```

### Example Response

```json
{
  "ok": true
}
```

---

## ESP32 Configuration

Before uploading the firmware, update the following parameters:

```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverUrl = "http://YOUR_RASPBERRY_PI_IP:5000/door";
```

---

## Arduino Libraries

Required libraries:

- WiFi.h
- HTTPClient.h

These libraries are included in the ESP32 Arduino Core package.

---

## Door Sensor Firmware

File:

```text
door_sensor.ino
```

Main Features:

- Detect door open event
- Detect door close event
- Connect to Wi-Fi
- Send HTTP POST requests to Raspberry Pi
- Real-time event reporting

---

## Integration with Smart Elder Guard

The camera system continuously monitors the environment using:

- YOLOv8 Person Detection
- ByteTrack Multi-Object Tracking
- MobileNetV3 Target Person Classification

When:

1. The target elderly person approaches the door area.
2. The target person disappears from the camera view.
3. A recent door-open event is reported by ESP32.

The system confirms an exit event and triggers cloud notification services.

---

## Advantages

- Low hardware cost
- Easy installation
- Real-time communication
- Improves exit-event reliability
- Reduces false alarms
- Seamless integration with Raspberry Pi and AWS IoT

---

## Future Improvements

- Wireless battery-powered door sensor
- MQTT communication instead of HTTP
- Multiple door monitoring
- Window sensor integration
- Additional indoor safety sensors

---

## Author

Yu-Ti Du  
Department of Electrical Engineering  
Yuan Ze University

2026

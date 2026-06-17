# Smart-Elder-Guard-AIoT-on-RaspberryPi5
Smart Elder Guard is an AIoT-based home monitoring system designed to detect nighttime exit events of dementia elderly people, which uesed Raspberry Pi, YOLOv8, ESP32 and AWS IoT.

## Overview

The project was developed on Raspberry Pi and evolved through multiple hardware and software iterations, including different camera platforms and sensor integrations:

* YOLOv8 Person Detection
* ByteTrack Multi-Object Tracking
* MobileNetV3 Target Person Classification
* ESP32 Door Sensor
* AWS IoT Core
* AWS Lambda
* LINE Messaging API

The system combines computer vision, edge computing, IoT communication, and cloud notification services to provide real-time monitoring and caregiver alerts.
When the target elderly person is detected leaving the house, the system automatically sends a real-time notification to caregivers through AWS cloud services and LINE.

## Motivation

Dementia patients may leave their homes unexpectedly, especially during nighttime hours.

Traditional monitoring methods often require continuous human supervision or expensive commercial solutions.

This project aims to provide a low-cost AIoT solution capable of:

* Detecting the target elderly person
* Monitoring movement near the entrance
* Identifying actual exit events
* Sending real-time notifications to caregivers

## System Architecture

### Continuous Vision-Based Monitoring

The camera is continuously active and serves as the main sensing source of the system.

```
Logitech Brio 500 / Raspberry Pi Camera Module 3 NoIR
        ↓
Raspberry Pi 5
        ↓
YOLOv8 Person Detection
        ↓
ByteTrack Multi-Object Tracking
        ↓
MobileNetV3 Target Person Classification
        ↓
Dynamic Voting Mechanism
        ↓
ROI-Based Exit Detection Logic
        ↓
AWS IoT Core
        ↓
AWS Lambda
        ↓
LINE Messaging API
        ↓
Caregiver Notification
```

### Door Sensor Assisted Verification
```
ESP32 Door Sensor
        ↓
HTTP Communication
        ↓
Raspberry Pi 5
        ↓
Door Event Verification
        ↓
Exit Detection Logic
```
The camera system continuously performs person detection and tracking.

The ESP32 door sensor serves as an auxiliary verification mechanism to confirm whether the door was actually opened, reducing false alarms caused by vision-only detection.

## Development History
### Version 1 – Logitech Brio 500

Hardware:

* Raspberry Pi 5
* Logitech Brio 500 USB Webcam

Features:

* YOLOv8 Person Detection
* ByteTrack Tracking
* MobileNetV3 Classification
* AWS IoT Notification

Limitations:

* Limited nighttime monitoring capability
* Exit detection relied solely on computer vision
### Version 2 – Raspberry Pi Camera Module 3 NoIR

Hardware:

* Raspberry Pi 5
* Raspberry Pi Camera Module 3 NoIR

Improvements:

* Better low-light performance
* Enhanced nighttime monitoring
* Native Raspberry Pi camera integration

Limitations:

* Exit-event verification still relied primarily on image analysis
### Version 3 – ESP32 Door Sensor Integration

Additional Hardware:

* ESP32 DevKit V1
* Magnetic Door Sensor

New Features:

* Door-open event detection
* Wi-Fi communication between ESP32 and Raspberry Pi
* Vision + sensor fusion verification

Benefits:

* Improved exit-event reliability
* Reduced false positive notifications

## Core Technologies
### Computer Vision
* YOLOv8
* ByteTrack
* OpenCV
### Person Identification
* MobileNetV3
* Dynamic Voting Mechanism
### Edge Computing
* Raspberry Pi 5
### IoT Communication
* ESP32
* HTTP Communication
* MQTT
### Cloud Services
* AWS IoT Core
* AWS Lambda
### Notification Service
* LINE Messaging API

## Hardware Components

## Software Requirements
* Python 3.13
* OpenCV
* Ultralytics YOLOv8
* Flask
* boto3
* AWS IoT SDK
* Arduino IDE

## Features
* Real-time Person Detection
* Multi-Person Tracking
* Target Elderly Identification
* Dynamic Voting Mechanism
* ROI-Based Exit Detection
* Door Sensor Verification
* AWS Cloud Integration
* LINE Notification Service
* Nighttime Monitoring

## Experimental Results
The system successfully:
* Detects the target elderly person in multi-person environments.
* Tracks movement using ByteTrack.
* Identifies exit events based on movement trajectory.
* Integrates door-open events through ESP32.
* Sends real-time notifications through AWS IoT and LINE.

## Future Work
* Fall Detection
* Abnormal Behavior Analysis
* Multi-Camera Tracking
* Activity History Analytics
* Infrared Night Monitoring
* Long-Term Health Monitoring
## Authors

* Yu-Ti Du (1120408)
* Yi-Ting Chiang (1120404)

Yuan Ze University, Department of Electrical Engineering

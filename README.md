# Smart-Elder-Guard-AIoT-on-RaspberryPi5

## Overview
Smart Elder Guard is an AIoT-based home monitoring system designed to detect nighttime exit events of dementia elderly people.

The system is deployed on Raspberry Pi 5 and integrates:

* YOLOv8 Person Detection
* ByteTrack Multi-Object Tracking
* MobileNetV3 Target Person Classification
* ESP32 Door Sensor
* AWS IoT Core
* AWS Lambda
* LINE Messaging API

When the target elderly person is detected leaving the house, the system automatically sends a real-time notification to caregivers through AWS cloud services and LINE.

## System Architecture

ESP32 Door Sensor
↓
Raspberry Pi 5
↓
YOLOv8 Person Detection
↓
ByteTrack Tracking
↓
MobileNetV3 Classification
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

## Hardware

* Raspberry Pi 5
* Logitech Brio 500 Webcam
* ESP32 DevKit V1
* Magnetic Door Sensor

## Software

* Python 3.13
* YOLOv8
* ByteTrack
* MobileNetV3
* Flask
* AWS IoT Core
* AWS Lambda
* Arduino IDE

## Features

* Real-time person detection
* Target elderly identification
* Multi-person tracking
* Dynamic voting mechanism
* Door sensor integration
* Cloud notification service
* LINE notification

## Authors

* Yu-Ti Du (1120408)
* Yi-Ting Chiang (1120404)

Yuan Ze University, Department of Electrical Engineering

# Lightweight Posture Monitor

A real-time posture monitoring system using MediaPipe and OpenCV to detect improper sitting posture through webcam input.

## Overview

This project uses computer vision techniques to analyze human posture by calculating neck and torso angles. It provides live feedback indicating whether the posture is correct or incorrect.

## Features

- Real-time posture detection using webcam
- Neck and torso angle calculation
- Good vs Bad posture classification
- Visual overlay with pose tracking
- Lightweight and efficient system

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## How It Works

1. Captures live video from webcam
2. Detects human body landmarks using MediaPipe Pose
3. Calculates:
   - Neck angle
   - Torso angle
4. Classifies posture:
   - Good Posture
   - Bad Posture
5. Displays real-time feedback on screen

## Installation

```bash
pip install -r requirements.txt

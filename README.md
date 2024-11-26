# CARLA Imitation Learning

This repository implements a pipeline for training an imitation learning model in the CARLA simulator. The goal is to collect driving data from the CARLA autopilot and train a convolutional neural network (CNN) to predict driving commands from front-camera images. This project serves as an entry point for building autonomous vehicle models based on supervised learning.

---

## Key Features
### Data Collection
- The `data_collect.py` script modifies CARLA's `manual_control.py` to enable data collection while driving in autopilot mode.
- Captures **front-camera RGB images** and **corresponding driving commands**.
- Saves the data in an organized structure, ready for training.

### Neural Network Architecture
- A custom-designed convolutional neural network (CNN) processes RGB images and predicts driving commands.
- The network features:
  - Five convolutional layers to extract spatial features from images.
  - Fully connected layers to map features to driving commands.
  - ReLU activation functions for non-linearity.

### Training Utilities
- PyTorch-based dataset and network definitions for seamless model training.
- A loss plotting script to monitor and visualize model performance during training.

---

## Neural Network Architecture

| Layer Type         | Parameters                              | Output Shape       | Description                                       |
|--------------------|-----------------------------------------|--------------------|---------------------------------------------------|
| **Input Layer**    | RGB Images (3 channels, 240x320 pixels)| (3, 240, 320)      | Input front-camera image from the CARLA simulator. |
| **Conv2D (1)**     | Filters: 24, Kernel: 5x5, Stride: 2     | (24, 118, 158)     | Extracts low-level spatial features.             |
| **ReLU Activation**| -                                       | (24, 118, 158)     | Non-linear activation for better feature learning.|
| **Conv2D (2)**     | Filters: 36, Kernel: 5x5, Stride: 2     | (36, 57, 77)       | Extracts mid-level spatial features.             |
| **ReLU Activation**| -                                       | (36, 57, 77)       |                                                   |
| **Conv2D (3)**     | Filters: 48, Kernel: 5x5, Stride: 2     | (48, 27, 37)       | Learns high-level spatial patterns.              |
| **ReLU Activation**| -                                       | (48, 27, 37)       |                                                   |
| **Conv2D (4)**     | Filters: 64, Kernel: 3x3, Stride: 2     | (64, 13, 18)       | Refines feature representations.                 |
| **ReLU Activation**| -                                       | (64, 13, 18)       |                                                   |
| **Conv2D (5)**     | Filters: 64, Kernel: 3x3, Stride: 1     | (64, 11, 16)       | Enhances spatial resolution further.             |
| **ReLU Activation**| -                                       | (64, 11, 16)       |                                                   |
| **Flatten**        | -                                       | (11264)            | Flattens spatial features into a 1D vector.      |
| **Linear (1)**     | Input: 11264, Output: 512               | (512)              | Reduces dimensionality, learning high-level concepts. |
| **ReLU Activation**| -                                       | (512)              |                                                   |
| **Linear (2)**     | Input: 512, Output: 100                 | (100)              | Further compression of features.                 |
| **ReLU Activation**| -                                       | (100)              |                                                   |
| **Linear (3)**     | Input: 100, Output: 50                  | (50)               | Captures finer details in the representation.    |
| **ReLU Activation**| -                                       | (50)               |                                                   |
| **Output Layer**   | Input: 50, Output: 9                    | (9)                | Predicts the driving command vector.             |

---

## Prerequisites

### Software
- **CARLA Simulator**: Tested on version 0.9.10.1
- **Python**: 3.8.


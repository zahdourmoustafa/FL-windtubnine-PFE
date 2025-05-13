# Federated Learning-based Gearbox Fault Prediction System (PRD)

## Overview

We propose a federated machine-learning system for wind turbine gearbox fault detection and mode classification. The system targets vibration data from multiple turbines (clients), each providing both **healthy** and **damaged** gearbox sensor readings. Using federated learning, clients collaboratively train a shared model **without sharing raw data**, preserving data privacy. This approach suits industrial IoT: each turbine site (client) keeps its sensor logs locally and only model updates are communicated to a central server. Our goal is to automatically detect sensor-level anomalies and identify gearbox failure modes (e.g. bearing overheating, gear scuffing) with high accuracy (>90%), using a PyTorch-based implementation (no TensorFlow or external FL frameworks). The design leverages accelerometer channels AN3–AN10 (vibration on ring gear, shafts and bearings) sampled at 40 kHz. By combining data from all clients, we aim to build a robust fault-detection model that generalizes across different turbines and fault scenarios.

## Data Description

Each client has two data files (healthy and damaged) in MATLAB `.mat` format. Each file contains **8 accelerometer channels (AN3–AN10)** and a shaft speed channel. Each sensor channel is a 1D time series of **2,400,000 samples** (60 seconds at 40 kHz). The sensors AN3–AN10 are mounted at key gearbox locations (see Table 1). The ground truth includes a **"healthy" vs "damaged"** label for each file, plus detailed **failure mode labels** for the damaged gearbox.

**Table 1. Sensor channels and locations:**  
| Sensor | Location / Component (radial direction) |
|--------|----------------------------------------------------|
| AN3 | Ring gear radial (6 o'clock position) |
| AN4 | Ring gear radial (12 o'clock) |
| AN5 | Low-speed stage sun gear (LS-SH) radial |
| AN6 | Intermediate-speed stage sun gear (IMS-SH) radial |
| AN7 | High-speed stage sun gear (HS-SH) radial |
| AN8 | HS-SH upwind bearing (radial on shaft housing) |
| AN9 | HS-SH downwind bearing (radial on shaft housing) |
| AN10 | Planetary carrier, downwind side radial |

**Table 2. Known gearbox failure components and modes:**  
| ID | Component | Failure Mode |
|----|-------------------------------------|--------------------------------------------------|
| 1 | HS-ST gear set (High-speed stage) | Scuffing |
| 2 | HS-SH downwind bearings | Overheating |
| 3 | IMS-ST gear set (Intermediate) | Fretting corrosion, scuffing, polishing wear |
| 4 | IMS-SH upwind bearing | Assembly damage, scuffing, dents |
| 5 | IMS-SH downwind bearings | Assembly damage, dents |
| 6 | Ring gear (annulus) or sun pinion | Scuffing and polishing; fretting corrosion |
| 7 | Planet carrier upwind bearing | Fretting corrosion |

**Table 3. Sensor-Failure Mode Mapping:**  

| Sensor   | Location / Component                   | Likely Failure Mode(s)                                               |
| -------- | -------------------------------------- | -------------------------------------------------------------------- |
| **AN3**  | Ring gear radial (6 o'clock)           | Scuffing, polishing, fretting corrosion                              |
| **AN4**  | Ring gear radial (12 o'clock)          | Scuffing, polishing, fretting corrosion                              |
| **AN5**  | LS-SH sun gear radial                  | Not directly mapped (Low-speed stage not listed)                     |
| **AN6**  | IMS-SH sun gear radial                 | Fretting corrosion, scuffing, polishing wear, assembly damage, dents |
| **AN7**  | HS-SH sun gear radial                  | Scuffing                                                             |
| **AN8**  | HS-SH upwind bearing                   | No direct match (downwind is mentioned)                              |
| **AN9**  | HS-SH downwind bearing                 | Overheating                                                          |
| **AN10** | Planetary carrier downwind side radial | Fretting corrosion                                                   |

## Preprocessing

1. **Windowing:** Segment each raw channel into fixed-length windows (e.g., 1-second = 40,000 samples). Each window retains all 8 sensor channels (`[8×40000]`).
2. **Normalization:** Zero-center and scale each window.
3. **Feature Extraction (optional):** Use FFT and statistical features like RMS, kurtosis, etc.

## Model Design

- Input: 8-channel window (`[8×40000]`).
- CNN layers to extract local features.
- LSTM for sequence learning.
- Output: binary or multi-class fault prediction.
- Use softmax or sigmoid + cross-entropy loss.

## Federated Learning Design

- Custom FedAvg without FL frameworks.
- Server broadcasts global model to clients.
- Clients train locally on 60% and validate on 10%.
- Clients send weights back; server aggregates.
- Repeat for 20–50 rounds.

## Training Strategy

- Each client: 60% train, 10% validation, 30% test.
- Local training: e.g., 5–10 epochs per round.
- Optimizer: Adam or SGD.
- Augmentation: Optional Gaussian noise or shifts.

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- Evaluate per sensor and globally.
- Target: >90% accuracy.

## User Interface

- Per-sensor status indicators (CLI or web).
- Mode correlation and time-series plots.
- Reports with confusion matrix and summaries.



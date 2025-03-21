# Federated Learning for Wind Turbine Gearbox Failure Prediction

## 📌 Project Overview
Wind turbines are critical in the transition to renewable energy, but gearbox failures lead to high maintenance costs and energy losses. This project aims to predict gearbox failures using Federated Learning (FL), ensuring data privacy while enabling collaborative model training across decentralized IoT sensors.

## 🎯 Objectives
- Develop a federated learning model that predicts gearbox failures with high accuracy.
- Ensure data privacy by training models on local IoT devices without sharing raw data.
- Implement a distributed learning framework that enables multiple wind farms to collaborate.
- Evaluate FL performance vs. centralized learning, focusing on accuracy, communication efficiency, and scalability.

## 📂 Dataset Overview

### 1️⃣ Data Source
- Collected from the National Renewable Energy Laboratory (NREL).
- Contains vibration condition monitoring data from wind turbine gearboxes.

### 2️⃣ Data Structure
The dataset is organized into two directories:
- **`data/damaged`**: Contains damaged gearbox files (`D1.mat`, `D2.mat` ).
- **`data/healthy`**: Contains healthy gearbox files (`H1.mat`, `H2.mat`).

#### Damaged Gearbox Files (`D1.mat`, `D2.mat`)
- **Sensors included**: AN3, AN4, AN5, AN6, AN7, AN8, AN9, AN10, Speed
- **Total Values per Sensor**: 2,400,000.
- **Torque**: Present only in damaged files.

#### Healthy Gearbox Files (`H1.mat`, `H2.mat`)
- **Sensors included**: AN3, AN4, AN5, AN6, AN7, AN8, AN9, AN10, Speed.
- **Total Values per Sensor**: 2,400,000.
- **Torque**: Missing in healthy files.

### 3️⃣ Sensor Descriptions
- **AN3, AN4**: Ring gear vibration (radial at 6 o'clock and 12 o'clock positions).
- **AN5**: Low-speed shaft (LS-SH) vibration (radial).
- **AN6**: Intermediate-speed shaft (IMS-SH) vibration (radial).
- **AN7**: High-speed shaft (HS-SH) vibration (radial).
- **AN8**: HS-SH upwind bearing vibration (radial).
- **AN9**: HS-SH downwind bearing vibration (radial).
- **AN10**: Carrier downwind radial vibration.
- **Speed**: Rotational speed of the high-speed shaft (HSS).


### **4. Actual Gearbox Damage**

The **"damaged" gearbox** experienced two oil-loss events in the field. It was later disassembled, and a detailed failure analysis was conducted [3].  
Table 5 summarizes the actual damage detected through vibration analysis.

---

### **Table 5: Actual Gearbox Damage Deemed Detectable through Vibration Analysis**

| Damage # | Component                        | Mode                                         |
| -------- | -------------------------------- | -------------------------------------------- |
| 1        | HS-ST gear set                   | Scuffing                                     |
| 2        | HS-SH downwind bearings          | Overheating                                  |
| 3        | IMS-ST gear set                  | Fretting corrosion, scuffing, polishing wear |
| 4        | IMS-SH upwind bearing            | Assembly damage, scuffing, dents             |
| 5        | IMS-SH downwind bearings         | Assembly damage, dents                       |
| 6        | Annulus/ring gear, or sun pinion | Scuffing and polishing, fretting corrosion   |
| 7        | Planet carrier upwind bearing    | Fretting corrosion                           |

---

## 🛠️ Data Preprocessing

### 1️⃣ Handling Large Data (2.4M values per sensor)
- **Downsampling**: Reduce frequency (e.g., take every 10th or 100th value) to improve efficiency.
- **Segmentation**: Split into smaller time windows (e.g., 1-second frames).

### 2️⃣ Normalization
- Apply Min-Max Scaling or Z-score Normalization to standardize data across sensors.

### 3️⃣ Handling Missing Torque in Healthy Data
- Set torque to zero for healthy data.
- Predict torque using other sensor values via regression.

### 4️⃣ Feature Extraction
- **Statistical Features**: Mean, Variance, RMS, Skewness, Kurtosis.
- **Frequency Domain Analysis**: Fast Fourier Transform (FFT) to detect anomalies.

### 5️⃣ Labeling
- `0` = Healthy Gearbox
- `1` = Damaged Gearbox

## 🤖 Machine Learning Model

### 🔹 CNN + LSTM Hybrid (Recommended)
**Why?**
- CNN extracts spatial patterns in vibration signals.
- LSTM captures long-term dependencies in time-series data.
- Balances accuracy & efficiency for real-world deployment.

**Model Structure**
- **CNN Layers**:
  - Extract features from vibration signals.
  - Detect spatial correlations.
- **LSTM Layers**:
  - Capture temporal relationships between sensor values.
  - Identify failure patterns before breakdowns.
- **Fully Connected Layer**:
  - Outputs probability of failure.

## 🌍 Federated Learning Strategy

### 1️⃣ Why Federated Learning?
- **Data Privacy**: Wind farms are geographically distributed, and sharing raw data may violate privacy or regulatory constraints.
- **Decentralized Training**: FL allows multiple wind farms to collaboratively train a global model without transferring sensitive raw data.
- **Scalability**: FL supports integration with IoT devices and edge computing, making it suitable for real-time failure prediction.

### 2️⃣ Federated Learning Workflow
**Step-by-Step Process**
1. **Model Initialization**:
   - A central server initializes a global model (e.g., CNN + LSTM architecture).
   - The model is distributed to all participating clients (wind farms or IoT devices).
2. **Local Training**:
   - Each client trains the model locally using its own dataset (e.g., H#.mat for healthy data and D#.mat for damaged data).
   - For example:
     - Client 1 might use H1.mat (healthy) and D1.mat (damaged).
     - Client 2 might use H2.mat (healthy) and D2.mat (damaged).
   - Local training ensures that raw sensor data never leaves the client's environment.
3. **Weight Sharing**:
   - After local training, clients send only the updated model weights (not raw data) to the central server.
   - This minimizes communication overhead and preserves data privacy.
4. **Aggregation**:
   - The central server aggregates the weights from all clients using Federated Averaging (FedAvg):
     - Compute the weighted average of the model parameters based on the number of samples each client used for training.
     - The aggregated weights form the updated global model.
5. **Model Deployment**:
   - The updated global model is redistributed to all clients for the next round of local training.
   - This iterative process continues until the model converges.
6. **Evaluation**:
   - Evaluate the global model on a centralized validation dataset (if available) or through decentralized testing across clients.
   - Metrics such as accuracy, F1-score, and communication efficiency are tracked to assess performance.

### 3️⃣ Challenges & Solutions
**Challenge 1: Non-IID Data**
- **Problem**: Sensor data from different wind farms may vary significantly due to environmental conditions (e.g., wind speed, temperature). This creates non-independent and identically distributed (non-IID) data, which can degrade FL performance.
- **Solution**:
  - Use personalized federated learning techniques, where each client fine-tunes the global model to better suit its local data distribution.
  - Implement data augmentation during preprocessing to reduce variability.

**Challenge 2: Communication Overhead**
- **Problem**: Frequent weight updates between clients and the server can lead to high communication costs, especially for large models like CNN + LSTM.
- **Solution**:
  - Use model compression techniques such as quantization or sparsification to reduce the size of transmitted weights.
  - Limit the frequency of weight updates by increasing the number of local training epochs before aggregation.

**Challenge 3: Heterogeneous Clients**
- **Problem**: Clients (e.g., IoT devices) may have varying computational capabilities, leading to uneven contributions to the global model.
- **Solution**:
  - Implement asynchronous federated learning, where clients update the server at their own pace without waiting for others.
  - Assign weights to clients based on their dataset size or computational power during aggregation.

### 4️⃣ Evaluation Metrics
- **Accuracy**: Measures overall correctness of failure prediction. Compare predicted vs. actual labels.
- **F1-Score**: Balances precision and recall for imbalanced datasets. Calculate harmonic mean of precision and recall.
- **Communication Efficiency**: Tracks data transfer volume during FL. Measure size of transmitted weights per round.
- **Model Convergence Time**: Evaluates how quickly the model learns. Track loss reduction over rounds.

## 🚀 Implementation Roadmap
1. **Data Preprocessing**
   - Load and downsample dataset.
   - Normalize and extract features.
   - Label the data for training.
2. **Model Training (Federated Learning)**
   - Set up PySyft or TensorFlow Federated for FL.
   - Train CNN + LSTM model on local clients.
   - Use FedAvg to aggregate models.
3. **Evaluation & Deployment**
   - Compare FL model vs. Centralized model.
   - Optimize for real-time failure prediction.
   - Deploy on IoT-based wind turbine monitoring systems.



   take al look at this pdf i want to add new feature to my code , which is when i test with unseen data , it should be tedect the sensors position damaged and also showing me the mode the damaged for example sensor damaged is an4 , the mode is overheating , 
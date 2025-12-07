# 🛰️ UWB Indoor Positioning using CNN-based NLoS Classification & ML-based Position Estimation

This repository contains the full implementation of a **UWB-based indoor positioning system** developed as part of my **Master’s thesis and extracted research paper**.  
The project focuses on:

- **NLoS/LoS classification using a novel CNN architecture**
- **Machine learning & ensemble learning for precise position estimation**
- **Signal preprocessing using Channel Impulse Response (CIR)**

![NLOS_LOS](NLOS_LOS.jpg)  
**Figure:** Difference of NLoS/LoS Conditions

All simulations, dataset preprocessing, CNN training, and ML-based positioning models are included in this repository.

---

## 📌 Project Overview

In UWB indoor positioning systems, **Non-Line-of-Sight (NLoS)** propagation introduces **positive ranging bias**, which significantly reduces localization accuracy.  
To address this issue, this project proposes:

1. **A deep CNN-based classifier** to distinguish **LoS vs NLoS** using **raw CIR data**
2. **Regression-based positioning using classical machine learning and ensemble learning**
3. **Performance evaluation using MAE, CDF, and real-vs-predicted plots**

The proposed CNN achieved over **92% classification accuracy** and significantly improved **positioning performance under NLoS conditions**.

**Note**: The UWB dataset used in this project is publicly available and can be accessed from the following official source:
[Dataset](https://log-a-tec.eu/uwb-ds.html)  
Due to file size limitations, the raw dataset is not included in this repository.

---

## 🧠 System Architecture

### 1️⃣ Data Preprocessing Pipeline  
Implemented in:

- `Create Dataset.py`

Main steps:
- Extracting **absolute values of complex CIR**
- Removing unnecessary columns
- Stacking and merging multi-anchor CIR measurements
- Building final **training/testing CSV datasets**

---

### 2️⃣ Novel CNN-Based LoS/NLoS Classification  
Implemented in:

- `CNN_New_Stracture.py`

In this work, a **novel 1D CNN architecture is proposed** specifically designed for **UWB Channel Impulse Response (CIR) signals** to achieve high-accuracy **LoS/NLoS classification**.

Key model structure:
- **1D Convolutional Neural Networks (Conv1D)** optimized for temporal CIR features
- **Max-Pooling layers** for dimensionality reduction
- **Fully Connected layers**:
  - FC1 = 128  
  - FC2 = 64 or 128
- **Dropout regularization**:
  - `0.5` (main optimized model)
  - `0.8` (overfitting analysis)

Output:
- Binary classification: **LoS / NLoS**


---

### 3️⃣ ML & Ensemble Position Estimation  
Implemented in:

- `positioning.py`

Models used:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Meta-Learners:
  - Linear Regression
  - Neural Network (MLP)

Techniques:
- **K-Fold Cross Validation**
- **Weighted Ensemble using inverse MAE**
- **Stacked Meta-Learning**
- **Execution time benchmarking**
- **Empirical CDF of positioning error**

---

## 📂 Repository Structure

├── requirements.txt

├── Create Dataset.py # CIR preprocessing & dataset generation

├── CNN_New_Stracture.py # CNN LoS/NLoS classifier

├── positioning.py # ML & Ensemble-based positioning

├── Report.pdf # Full Master’s thesis

├── Paper.pdf # Extracted research paper

└── README.md

---

## 📌 Citation

If you use any part of this code, methodology, or results in your own research, **please cite the following paper**:

🔗 https://ieeexplore.ieee.org/abstract/document/10533361


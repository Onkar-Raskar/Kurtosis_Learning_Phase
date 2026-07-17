# Anomaly Detection 🏭🔊

## Overview
This project explores predictive maintenance through acoustic anomaly detection. By analyzing the soundscapes of industrial machinery (Pumps, Fans, and Valves), the goal is to identify mechanical failures using unsupervised machine learning and autoencoder architectures.

## The Dataset
The models are trained and evaluated on segments of the **MIMII (Malfunctioning Industrial Machine Investigation and Inspection) Dataset**.
*   **Normal Audio:** Baseline humming and clicking of healthy machines.
*   **Abnormal Audio:** Recordings of specific mechanical failures (e.g., chipped fan blades, grinding pump bearings, leaking valves).

## Model Performance & Baselines

Rigorous, leak-free testing yielded the following baseline metrics using global-mean features and 1D Autoencoders:

| Machine Type | Sound Profile | Recall | Precision | F1 Score | 
| :--- | :--- | :--- | :--- | :--- |
| **Pump** | Continuous | 97.2% | 90.8% | **~0.93** |
| **Fan** | Continuous | 97.7% | 82.4% | **~0.89** |
| **Valve** | Impulsive | 90.7% | 50.7% | **~0.65** |


## Getting Started

### To Test it Yourself 
First Download the Dataset from website : https://zenodo.org/records/3384388

Download files named 6_dB_fan.zip, 6_dB_pump.zip,  6_dB_pump.zip

extract these zip files in folder data ensuring directory structure as
``` 
├── data/
│   ├── fan/id_00/
│   │   ├── normal/
│   │   └── abnormal/
│   ├── pump/id_00/
│   │   ├── normal/
│   │   └── abnormal/
│   └── valve/id_00/
│       ├── normal/
│       └── abnormal/
└── Autoencoder.py
```

### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install torch torchvision torchaudio tensorflow librosa scikit-learn numpy pandas matplotlib seaborn jupyter
```
now ready to run Autoencoder.py

# Bearings Classifier

A machine learning project for bearing condition monitoring and fault detection using vibration signal analysis.

## Overview

This mini-project focuses on analyzing vibration signals from bearings (CRWU dataset) to distinguish between healthy and faulty conditions. It includes preprocessing of vibration data, visualization of time-domain signals, and serves as the foundation for implementing machine learning and deep learning models for bearing fault diagnosis.

## Project Structure

```
Bearings_Classifier/
│
├── Data/
│   ├── normal.mat
│   └── faulty.mat
│
├── Model/
│   ├── model.py
│   └── plot.py
|
├── Output/
    ├── output.png
    ├── plot.png
```

## Features

- Load MATLAB (`.mat`) vibration datasets
- Plot healthy and faulty bearing vibration signals
- Time-domain signal visualization
- Foundation for frequency-domain analysis (FFT)
- Ready for feature extraction and machine learning models

## Technologies Used

- Python
- NumPy
- SciPy
- Matplotlib
- sckit-learn

## Dataset

The project uses MATLAB (`.mat`) files containing vibration signals:

- `normal.mat` – Healthy bearing data
- `faulty.mat` – Faulty bearing data

## Future Work

- Fast Fourier Transform (FFT)
- Feature extraction
- CNN-based fault classification
- Temporal Convolutional Networks (TCN)

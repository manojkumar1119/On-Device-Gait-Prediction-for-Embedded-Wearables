#On-Device-Gait-Prediction-for-Embedded-Wearables

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [Dataset](#dataset)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Architectures](#model-architectures)
- [Quantization Strategy](#quantization-strategy)
- [Results & Performance](#results--performance)
- [Model Selection Rationale](#model-selection-rationale)
- [Deployment](#deployment)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Specifications](#technical-specifications)
- [Future Work](#future-work)
---

## Overview

This project demonstrates the feasibility of deploying sophisticated machine learning models for **real-time human gait trajectory prediction** on resource-constrained embedded microcontrollers. By leveraging **TensorFlow Lite (TFLite)** with **INT8 quantization**, we achieve high-accuracy biomechanical forecasting with minimal latency and memory footprint, making it suitable for edge devices like the **Arduino Nano BLE Sense**.

### Problem Statement

Traditional gait analysis systems require computationally expensive hardware and cannot operate in real-time on embedded devices. This project addresses the challenge of bringing ML-powered gait prediction to the edge, enabling applications in:

- **Smart Prosthetics**: Real-time adaptive control
- **Exoskeletons**: Predictive assistance algorithms
- **Wearable Health Monitors**: Continuous gait analysis
- **Rehabilitation Devices**: Patient movement tracking

### Solution

Optimized neural network architecture was trained on the **HuGaDB (Human Gait Database)** and converted them to fully quantized TFLite models that can run efficiently on microcontrollers with limited RAM and flash memory.

##  Key Features

- **Edge-Optimized ML Models**: Lightweight architectures designed specifically for microcontrollers
- **INT8 Quantization**: Full integer quantization reducing model size by ~75% with minimal accuracy loss
- **Real-Time Inference**: Low-latency predictions suitable for time-critical applications
- **Memory Efficient**: Final model size of only **15.5 KB**, fitting comfortably in microcontroller flash
- **High Accuracy**: Achieves **90.88%** file-level accuracy post-quantization
- **Modular Pipeline**: Reusable data processing and model training components
- **Multiple Architectures**: Comparative analysis of MLP and DSCNN approaches

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Conda**: For environment management
- **Arduino IDE**:For microcontroller deployment

### Step 1: Create Conda Environment

```bash
# Create environment from specification file
conda env create -f environment.yml

# Activate the environment
conda activate gait-prediction
```

### Step 2: Verify Installation

```bash
# Check Python version
python --version

# Check TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

Expected output:
```
Python 3.8.x
TensorFlow version: 2.18.0
```

### Step 3: Download Dataset

```bash
# Run the dataset fetch script
python .fetch_dataset.py --extract_to dataset_v2
```

This will:
- Download the **HuGaDB v2** dataset (~89 MB ZIP file)
- Extract **364 data files** to `./dataset_v2/`
- Verify data integrity

---

## Dataset

### HuGaDB (Human Gait Database) v2

**Source**: [HuGaDB Repository](https://github.com/romanchereshnev/HuGaDB)

**Description**: A comprehensive database containing inertial sensor measurements from multiple body locations during various walking activities.

#### Dataset Specifications

- **Total Files**: 364 CSV files
- **Subjects**: Multiple participants
- **Activities**: Normal walking, fast walking, slow walking, incline/decline walking
- **Sensors**: 
  - Accelerometers (3-axis)
  - Gyroscopes (3-axis)
  - Placed on left/right shank, thigh, and foot
- **Sampling Rate**: 100 Hz
- **File Size**: ~89 MB (compressed)

#### Selected Signals

For this project,  **4-signal subset** is used from the left limb:

| Signal | Description | Axis | Sensor Location |
|:-------|:------------|:-----|:----------------|
| `acc_ls_x` | Acceleration | X-axis | Left Shank |
| `acc_ls_z` | Acceleration | Z-axis | Left Shank |
| `gyro_ls_y` | Angular Velocity | Y-axis | Left Shank |
| `gyro_lf_y` | Angular Velocity | Y-axis | Left Foot |

---

##  Data Processing Pipeline

The preprocessing pipeline is implemented in `hugadb_v2_pipeline.py` and consists of three main stages:

### Stage 1: Signal Selection

```python
selected_signals = ['acc_ls_x', 'acc_ls_z', 'gyro_ls_y', 'gyro_lf_y']
```

Extracts only the relevant sensor channels from the raw dataset, reducing input dimensionality.

### Stage 2: Moving Average Filtering

**Filter Type**: 3-tap moving average filter

**Purpose**: 
- Noise reduction
- Signal smoothing
- Preserving temporal characteristics

**Implementation**:
```python
# Convolution-based moving average
window = np.ones(3) / 3
filtered_signal = np.convolve(signal, window, mode='valid')
```

### Stage 3: Windowing

**Parameters**:
- **Window Size**: Configurable (typically 50-100 samples)
- **Stride**: Configurable (typically 10-25 samples for overlap)
- **Output**: (X, Y) pairs where X is the input window, Y is the corresponding label

**Process**:
1. Slide a fixed-size window across the time series
2. Extract features from each window
3. Assign labels based on gait phase
4. Create training/validation/test splits

**Example**:
```
Time Series: [t0, t1, t2, ..., tn]
Window 1: [t0:t50] → Label: Phase A
Window 2: [t10:t60] → Label: Phase A
Window 3: [t20:t70] → Label: Phase B
...
```

---

## Model Architectures

We developed and compared two distinct architectural approaches:

### 1. Multi-Layer Perceptron (MLP)

**Model ID**: `mlp_cls_32-16`

**Architecture**:
```
Input Layer (Flattened Window)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dense Layer (16 units, ReLU)
    ↓
Output Layer (Softmax)
```

**Characteristics**:
- **Total Parameters**: ~10,000
- **Layers**: 3 fully connected layers
- **Activation**: ReLU (hidden), Softmax (output)
- **Advantages**: 
  - Minimal parameter count
  - Fastest inference time
  - Smallest model size
  - Easy to optimize for edge devices

### 2. Depthwise Separable CNN (DSCNN)

**Model IDs**: `dscnn_cls_b{16,24,32}_res`

**Architecture**:
```
Input Layer (Window with Channel Dimension)
    ↓
Depthwise Separable Conv Block 1
    ↓ (+ Residual Connection)
Depthwise Separable Conv Block 2
    ↓ (+ Residual Connection)
Global Average Pooling
    ↓
Dense Layer
    ↓
Output Layer (Softmax)
```

**Depthwise Separable Convolution**:
- Splits standard convolution into:
  1. **Depthwise**: Spatial filtering per channel
  2. **Pointwise**: 1×1 convolution for channel mixing
- **Benefit**: Reduces parameters by ~8-9× vs. standard convolutions

**Variants**:

| Model | Base Filters | Parameters | Characteristics |
|:------|:-------------|:-----------|:----------------|
| `dscnn_cls_b16_res` | 16 | ~50K | Lightweight CNN |
| `dscnn_cls_b24_res` | 24 | ~100K | Balanced model |
| `dscnn_cls_b32_res` | 32 | ~175K | Highest capacity |


---

##  Quantization Strategy

### Why Quantization?

**Goal**: Deploy models on microcontrollers with:
- Limited flash memory (< 1 MB)
- Limited RAM (< 256 KB)
- No floating-point unit (FPU)

### Full INT8 Quantization

**Process**:
1. **Train in FP32**: Normal training with float32 precision
2. **Representative Dataset**: Create calibration dataset from training data
3. **Quantization-Aware Calibration**: Determine optimal quantization parameters
4. **Convert to INT8**: All weights, activations, and operations use 8-bit integers

**TensorFlow Lite Converter Configuration**:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()
```

### Benefits

| Metric | FP32 Model | INT8 Model | Improvement |
|:-------|:-----------|:-----------|:------------|
| Model Size | 62 KB | 15.5 KB | **75% reduction** |
| Inference Speed | Baseline | 2-4× faster | **2-4× speedup** |
| Memory Usage | Baseline | 4× lower | **75% reduction** |
| Power Consumption | Baseline | 3-5× lower | **60-80% reduction** |

---

## Results & Performance

### Comprehensive Model Comparison

| Model | Architecture | Float32 Acc | INT8 Acc | TFLite Size | Efficiency Score | Inference Time* |
|:------|:-------------|:------------|:---------|:------------|:-----------------|:----------------|
| **mlp_cls_32-16**  | 3-layer MLP | 91.12% | **90.88%** | **15.5 KB** | **0.0806** | ~2 ms |
| dscnn_cls_b16_res | DSCNN-16 | 95.80% | 95.52% | 25.0 KB | 0.0351 | ~5 ms |
| dscnn_cls_b24_res | DSCNN-24 | 95.52% | 95.24% | 34.5 KB | 0.0317 | ~7 ms |
| dscnn_cls_b32_res | DSCNN-32 | 94.12% | 93.76% | 45.5 KB | 0.0282 | ~10 ms |

*Estimated inference time on Arduino Nano BLE Sense (64 MHz Cortex-M4)

**Efficiency Score**: Calculated as `Accuracy / Size_KB` (higher is better)

### Key Observations

1. **Quantization Impact**: Minimal accuracy degradation (< 0.5% average) after INT8 quantization
2. **Size-Accuracy Trade-off**: MLP achieves best efficiency score despite lower raw accuracy
3. **DSCNN Performance**: Higher accuracy but 1.6-3× larger model size
4. **Real-time Capable**: All models meet real-time requirements (< 10 ms inference)

### Accuracy Metrics Explained

**File-Level Accuracy**: 
- Prediction aggregation over all windows in a single recording
- Majority voting used to determine final file prediction
- More robust metric for real-world deployment

---

## Model Selection Rationale

### Selected Model: `mlp_cls_32-16`

**Decision Factors**:

1. **Memory Constraints** (Weight: 40%)
   - Arduino Nano BLE Sense has 1 MB flash
   - Need room for bootloader, application code, and TFLite runtime
   - 15.5 KB model leaves ample space: ✅

2. **Inference Latency** (Weight: 30%)
   - Target: < 10 ms for real-time gait prediction
   - MLP achieves ~2 ms: ✅

3. **Accuracy Requirement** (Weight: 25%)
   - Minimum threshold: 85% for practical deployment
   - MLP achieves 90.88%: ✅

4. **Power Efficiency** (Weight: 5%)
   - Fewer operations = lower power consumption
   - Critical for battery-powered wearables: ✅

### Trade-off Analysis

```
DSCNN Models:
✅ Higher accuracy (+4-5%)
❌ 1.6-3× larger size
❌ 2.5-5× slower inference
❌ Higher power consumption

MLP Model:
✅ Smallest size (15.5 KB)
✅ Fastest inference (~2 ms)
✅ Lowest power draw
✅ Still high accuracy (90.88%)
✓ Best for production deployment
```

**Conclusion**: The MLP model provides the optimal balance for embedded deployment where resource constraints are paramount.

---

##  Deployment

### Arduino Nano BLE Sense Setup

**Hardware Requirements**:
- Arduino Nano 33 BLE Sense
- USB cable for programming
- (Optional) Battery for standalone operation

**Software Requirements**:
- Arduino IDE 2.0+
- TensorFlow Lite Micro library

### Deployment Steps

1. **Convert Model to C Array**:
```bash
xxd -i output/mlp_cls_32-16_int8.tflite > model_data.cc
```

2. **Arduino Sketch Structure**:
```cpp
#include <TensorFlowLite.h>
#include "model_data.h"

// Allocate memory for model
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  // Initialize TFLite interpreter
  // Load model
  // Allocate tensors
}

void loop() {
  // Read sensor data
  // Preprocess (moving average, windowing)
  // Run inference
  // Process predictions
}
```

3. **Flash to Device**:
   - Connect Arduino via USB
   - Upload sketch through Arduino IDE
   - Monitor serial output for predictions

### Expected Performance

- **RAM Usage**: ~20 KB (model + tensors)
- **Flash Usage**: ~50 KB (model + code)
- **Inference Time**: 2-3 ms per window
- **Power Draw**: ~5 mA during inference

---

##  Usage

### Training New Models

```bash
# Open Jupyter notebook
jupyter notebook project1.ipynb

# Or run as Python script
python -m jupyter nbconvert --to script project1.ipynb
python project1.py
```

### Running the Complete Pipeline

```python
import hugadb_v2_pipeline as pipeline

# Load and preprocess data
X_train, y_train = pipeline.load_and_preprocess(
    data_dir='./data/dataset_v2',
    signals=['acc_ls_x', 'acc_ls_z', 'gyro_ls_y', 'gyro_lf_y'],
    window_size=50,
    stride=10
)

# Train model
model = pipeline.create_mlp_model(input_shape=X_train.shape[1:])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Convert to TFLite
tflite_model = pipeline.convert_to_tflite_int8(model, X_train)

# Save model
with open('output/model_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

##  Project Structure

```
On-Device-Gait-Prediction-for-Embedded-Wearables/
│
│
│   └── fetch_dataset.py           # Dataset download script
│
│   └── project1.ipynb             # Main Jupyter notebook
│
│   └── hugadb_v2_pipeline.py      # Data processing utilities
│
├── arduino/                       # Arduino deployment code
│   ├── gait_prediction.ino
│   └── model_data.cc
│
├── environment.yml                # Conda environment spec
├── requirements.txt               # Pip requirements
├── README.md                      # This file
```

---

##  Technical Specifications

### Model Details

**MLP (mlp_cls_32-16)**:
```
Input Shape: (200,)  # 4 signals × 50 timesteps
Layer 1: Dense(32, activation='relu')
Layer 2: Dense(16, activation='relu')
Layer 3: Dense(num_classes, activation='softmax')
Total Parameters: ~10,240
Trainable Parameters: ~10,240
```

**DSCNN (dscnn_cls_b16_res)**:
```
Input Shape: (50, 4, 1)  # timesteps × signals × channels
DepthwiseSeparableConv1D(16, kernel=3)
BatchNormalization + ReLU
DepthwiseSeparableConv1D(16, kernel=3)
Residual Add
GlobalAveragePooling1D
Dense(num_classes, activation='softmax')
Total Parameters: ~52,000
```

### Hyperparameters

| Parameter | Value | Notes |
|:----------|:------|:------|
| Learning Rate | 0.001 | Adam optimizer |
| Batch Size | 32 | Balanced for memory/convergence |
| Epochs | 50 | With early stopping |
| Dropout | 0.2 | Applied to DSCNN models |
| L2 Regularization | 0.0001 | Weight decay |
| Window Size | 50 | 0.5 seconds at 100 Hz |
| Stride | 10 | 80% overlap |

##  Future Work

### Short-term Enhancements

1. **Model Optimization**
   - Pruning for further size reduction
   - Knowledge distillation from DSCNN to MLP
   - Dynamic quantization exploration

2. **Feature Engineering**
   - Frequency domain features (FFT)
   - Statistical features (mean, std, skewness)
   - Time-frequency representations (wavelet transforms)

3. **Deployment Improvements**
   - Real-time visualization dashboard
   - Bluetooth data transmission
   - Battery optimization strategies

### Long-term Research Directions

1. **Advanced Temporal Modeling**
   - Optimized LSTM/GRU architectures for TFLite
   - Temporal Convolutional Networks (TCN)
   - Transformer-based models with extreme quantization

2. **Multi-Sensor Fusion**
   - Incorporate pressure sensors
   - EMG signal integration
   - Camera-based pose estimation fusion

3. **Adaptive Control Systems**
   - **Smart Prosthetics**: Real-time gait phase detection for adaptive ankle control
   - **Exoskeletons**: Predictive assistance timing for energy-efficient walking
   - **Rehabilitation**: Personalized gait training with real-time feedback

4. **Personalization**
   - Few-shot learning for user adaptation
   - Transfer learning from general to personalized models
   - On-device continual learning

5. **Clinical Applications**
   - Parkinson's disease gait monitoring
   - Fall risk assessment
   - Post-surgery recovery tracking

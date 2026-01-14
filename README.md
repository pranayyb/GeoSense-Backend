# GeoSense-Backend

A robust API service for advanced land boundary detection using the Segment Anything Model (SAM), offering both single and bulk prediction capabilities with high accuracy rates of 75-90%.

## Table of Contents

- [Overview](#overview)
- [Sample Outputs](#sample-outputs)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Technical Dependencies](#technical-dependencies)
  - [Installation](#installation)
  - [Running the Service](#running-the-service)
- [API Usage](#api-usage)
  - [Single Image Prediction](#single-image-prediction)
  - [Bulk Image Prediction](#bulk-image-prediction)
- [Technical Implementation](#technical-implementation)
  - [SAM Model Configuration](#sam-model-configuration)
- [Development](#development)
- [Licensing](#licensing)
- [Acknowledgements](#acknowledgements)

## Overview

GeoSense-Backend is an advanced machine learning system designed for precise land boundary detection. The system utilizes the Segment-Anything Model (SAM) architecture to achieve highly accurate boundary detection in geospatial data. Through integration with Roboflow for data annotation, the system provides comprehensive land segmentation capabilities with documented accuracy rates of 75-90%.

## Sample Outputs

![Land Boundary Detection](https://github.com/user-attachments/assets/f7721c51-5965-40e0-a6ce-ca84b6179974)

![Segmented Image](https://github.com/user-attachments/assets/aa9aaa2b-8853-474f-8164-308bd0fcec0e)

![Visualization Example](https://github.com/user-attachments/assets/f5243ae4-f2fa-43d0-8f78-b7f07f15b1a2)

## Key Features

- Advanced Land Boundary Detection: Implementation of deep learning algorithms for precise boundary identification
- Optimized Segment-Anything Model: Custom-tuned SAM implementation specifically adapted for land imagery analysis
- Comprehensive Training Dataset: Extensive collection of annotated land boundary imagery
- Efficient Processing: Optimized for high-throughput analysis of geo-spatial data
- Bulk Upload Support: Process multiple images simultaneously for efficient batch analysis
- SVG Conversion: Automatic conversion of predictions to scalable vector graphics

## Directory Structure

```
├── LICENSE                                   # Legal documentation
├── README.md                                 # Project documentation
├── api/                                      # API endpoints for prediction
│   ├── bulk.py                               # Handles bulk image prediction
│   ├── img.py                                # Single image prediction handler
│   └── svg.py                                # Converts predictions to SVG
├── config.py                                 # Configuration and constants
├── main.py                                   # Entry point for the backend service
├── notebooks/                                # Jupyter notebooks for model development
│   ├── Segment-Boundary-Model-2.ipynb        # Training and evaluation
│   ├── image_predictor_example.ipynb         # Usage and example outputs
│   └── sam_2_2.ipynb                         # SAM tuning and optimization
├── requirements.txt                          # Project dependencies
├── utils.py                                  # Utility functions
└── weights/                                  # Pretrained model weights
    └── sam_vit_h_4b8939.pth                  # SAM model checkpoint
```

## Getting Started

### Technical Dependencies

The system relies on the following primary components:

- opencv-python: Image processing and data preprocessing
- torch: Deep learning framework implementation
- torchvision: Computer vision model support
- fastapi: API framework implementation
- uvicorn: ASGI server implementation
- segment_anything: SAM model integration
- supervision: Segmentation and visualization toolkit
- python-multipart: FastAPI file upload management
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository

   ```bash
   git clone https://github.com/NeonKazuha/GeoSense-Backend.git
   cd GeoSense-Backend
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Download the SAM model weights and place them in the `weights/` directory
   - The default configuration expects `sam_vit_h_4b8939.pth`
   - This will be downloaded once you run the main file

### Running the Service

Start the backend service:

```bash
python main.py
```

By default, the API will be available at `http://localhost:9000`.

## API Usage

### Single Image Prediction

```
POST /api/img
```

Request body should contain the image file. The endpoint processes individual images for land boundary detection and returns prediction results.

### Bulk Image Prediction

```
POST /api/bulk
```

Request body should contain multiple image files. This endpoint is optimized for high-throughput processing of multiple geospatial images simultaneously, improving efficiency for large-scale analysis tasks.

## Technical Implementation

### SAM Model Configuration

The implementation of the Segment Anything Model includes the following specifications:

- Dataset Configuration: High-resolution land boundary imagery with professional annotations
- Model Architecture: Vision Transformer (ViT) backbone with enhanced encoder-decoder segmentation pipeline
- Training Parameters:
  - Batch size range: 16-64
  - Optimizer: AdamW
  - Learning rate schedule: Cosine Annealing
  - Gradient Clipping: 1.0
  - Loss functions: Combined Dice Loss and Focal Loss
- Evaluation Metrics: IoU, Dice Score, and mAP for comprehensive performance assessment

The system implements domain-specific augmentation techniques to optimize the mask decoder for terrain feature detection.

## Development

Explore the Jupyter notebooks in the `notebooks/` directory for examples on:

- Training and evaluating segmentation models (`Segment-Boundary-Model-2.ipynb`)
- Using the image predictor (`image_predictor_example.ipynb`)
- Tuning the SAM model (`sam_2_2.ipynb`)

## Licensing

This project is distributed under the Apache License 2.0. Reference the LICENSE file for complete terms and conditions.

## Acknowledgements

This project uses the Segment Anything Model (SAM) developed by Meta AI Research and integrates with Roboflow for data annotation.

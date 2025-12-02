# KenalBatik: AI Motif Scanner

KenalBatik is a full-stack web application designed to identify and classify traditional Indonesian Batik motifs using Artificial Intelligence. The system utilizes a deep learning model to analyze uploaded images and provide accurate predictions regarding the specific Batik pattern.

## Project Overview

The application consists of a modern web interface for users to upload images and a robust backend API that handles image processing and inference. The core technology relies on a pre-trained MobileNetV2 neural network, fine-tuned for Batik motif classification.

## Features

- **Image Analysis**: Users can upload images of Batik cloth for instant analysis.
- **AI Classification**: Utilizes PyTorch and MobileNetV2 to classify motifs with confidence scores.
- **Responsive Design**: A modern frontend interface built with Next.js.
- **Fast API**: High-performance backend powered by FastAPI.

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Machine Learning**: PyTorch, Torchvision
- **Model**: MobileNetV2 (Pre-trained and Fine-tuned)
- **Image Processing**: Pillow (PIL)

### Frontend
- **Framework**: Next.js
- **Language**: TypeScript
- **Styling**: CSS Modules / Global CSS

## Project Structure

- `backend/`: Contains the FastAPI application, model weights (`batik_model_v1.pth`), and training scripts.
- `frontend/`: Contains the Next.js web application source code.

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

## License


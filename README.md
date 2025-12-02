# KenalBatik: AI Motif Scanner

KenalBatik is a full-stack web application designed to identify, classify, and educate users about traditional Indonesian Batik motifs using Artificial Intelligence. The system utilizes a deep learning model to analyze uploaded images, provide accurate predictions, and offer rich cultural context including historical origins and geographical data.

## Project Overview

The application bridges the gap between traditional heritage and modern technology. It features a responsive web interface for users to upload images and a robust backend API that handles image processing, inference, and data management. The core technology relies on a pre-trained MobileNetV2 neural network, fine-tuned specifically for Batik motif classification.

## Key Features

### AI Vision & Explainability
The application goes beyond simple classification by integrating **Grad-CAM (Gradient-weighted Class Activation Mapping)**. This technology visualizes the "attention" of the AI model, generating a heatmap overlay that highlights exactly which parts of the Batik pattern influenced the prediction. This makes the AI's decision-making process transparent and interpretable for the user.

### Interactive Culture Map
To enhance the educational value, KenalBatik includes an **Interactive Culture Map**. When a Batik pattern is identified, the application dynamically displays its historical origin on a map of Indonesia. This feature is powered by Leaflet and provides users with a geographical context for the cultural heritage they are exploring.

### Feedback Loop & Data Safety
The system includes a robust **Feedback Loop** to continuously improve the model. Users can correct misclassified predictions via the interface. To ensure data integrity and prevent "data poisoning," these user-submitted corrections are not immediately added to the training set. Instead, they are securely routed to a `pending_review` directory for manual verification before being integrated into the dataset.

### Geolocation API
The backend API has been enhanced to return precise geolocation data (Latitude, Longitude, and Origin Name) for each identified Batik class. This data powers the frontend map and provides a foundation for future location-based features.

## System Optimizations

### CPU-Bound Inference Optimization
The backend has been refactored to optimize for CPU-based inference. The prediction endpoint is implemented as a synchronous function, allowing FastAPI to offload the heavy PyTorch computations to a separate thread pool. This prevents the main event loop from being blocked, ensuring the server remains responsive even during intensive analysis tasks.

### LRU Caching
To further improve performance, the system implements **Least Recently Used (LRU) Caching**. The application calculates an MD5 hash of every uploaded image. If an identical image is uploaded again, the system retrieves the prediction result directly from memory, bypassing the expensive AI inference process entirely.

### Model Quantization
For deployment efficiency, the MobileNetV2 model has been compressed using **Static Quantization**. This process reduces the model size by approximately 4x (from ~8.8MB to ~2.2MB) with minimal impact on accuracy, making it significantly faster and more memory-efficient for CPU inference.

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Machine Learning**: PyTorch, Torchvision
- **Model**: MobileNetV2 (Quantized)
- **Image Processing**: Pillow (PIL)
- **Utilities**: Hashlib (Caching), Shutil (File Operations)

### Frontend
- **Framework**: Next.js
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Mapping**: Leaflet, React Leaflet
- **State Management**: React Hooks

## Project Structure

- `backend/`: Contains the FastAPI application, model weights (`batik_model_v1.pth`), quantization scripts, and temporary storage.
- `frontend/`: Contains the Next.js web application source code and components.

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

This project is open-source and available for educational purposes.

# KenalBatik: End-to-End MLOps Computer Vision Platform

![Next.js](https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)

A full-stack AI platform that identifies 20 distinct Indonesian Batik motifs using a custom MobileNetV2 model. The system features a Human-in-the-Loop reinforcement learning pipeline, Grad-CAM interpretability, and edge-optimized inference.

## Key Technical Features

- 🧠 **Custom AI Engine**: Transfer Learning on MobileNetV2 optimized for Apple Silicon (MPS).
- 🛡️ **Self-Healing Dataset (MLOps)**: Implemented a feedback loop where user corrections are sent to a security 'Quarantine' folder (`pending_review`) to prevent data poisoning before retraining.
- 🔍 **Explainable AI (XAI)**: Integrated Grad-CAM heatmaps to visualize neural network attention layers.
- ⚡ **Performance Engineering**: 400% model compression via Int8 Quantization and LRU Caching for sub-100ms inference.
- 🗺️ **Interactive Geolocation**: Dynamic mapping of pattern origins using Leaflet.js.

## Tech Stack

| Component | Technologies |
|-----------|--------------|
| **Frontend** | Next.js 14 (App Router), Tailwind CSS, Framer Motion, Leaflet |
| **Backend** | FastAPI, PyTorch, NumPy, Pillow |
| **Infrastructure** | Docker Compose, Git Flow |

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+

### 1. Clone the Repository
```bash
git clone https://github.com/grahanurdian/KenalBatik.git
cd KenalBatik
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:3000`.

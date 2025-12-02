from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
from gradcam import BatikGradCAM
import torch
from torchvision import models, transforms
import random
import shutil
import os
import uuid
from pydantic import BaseModel

app = FastAPI()

# Ensure temp_uploads exists
os.makedirs("temp_uploads", exist_ok=True)
# Ensure dataset exists (optional, but good practice if we are moving files there)
os.makedirs("dataset", exist_ok=True)

# --- 1. CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD THE BRAIN ---
# Detect hardware: Uses M1 GPU locally, but switches to CPU inside Docker automatically
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the Classes EXACTLY as your folder names appear (Alphabetical Order)
CLASS_NAMES = [
    "Bali", "Betawi", "Celup", "Cendrawasih", "Ceplok", 
    "Ciamis", "Garutan", "Gentongan", "Kawung", "Keraton", 
    "Lasem", "Mega Mendung", "Parang", "Pekalongan", "Priangan", 
    "Sekar Jagad", "Sidoluhur", "Sidomukti", "Sogan", "Tambal"
]

# Load the model structure
model = models.mobilenet_v2(weights=None)
# Reshape the output layer to match our 5 classes
model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASS_NAMES))

# Load the weights
try:
    state_dict = torch.load("batik_model_v1.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Lock the model for inference
    print(f"SUCCESS: Loaded Batik Model on {device}")
except FileNotFoundError:
    print("WARNING: 'batik_model_v1.pth' not found. App will crash if you try to analyze.")
    model = None

# Preprocessing pipeline (Must match what you used in train.py)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. KNOWLEDGE BASE ---
# --- 3. KNOWLEDGE BASE ---
BATIK_INFO = {
    "Bali": { 
        "description": "Bali batik often features natural elements like frangipani flowers, birds, and daily life, mixing traditional patterns with modern, vibrant colors.",
        "pattern": "Mix of traditional motifs with modern, free-form artistic expressions, often featuring bright colors and natural imagery like flowers, birds, and dancers."
    },
    "Betawi": {
        "description": "Originating from Jakarta, Betawi batik features bright colors and motifs inspired by local culture like Ondel-ondel, the Ciliwung river, and tumpal patterns.",
        "pattern": "Bright, contrasting colors with motifs of Jakarta's icons (Monas, Ondel-ondel), flora, and fauna, often arranged in a tumpal (triangle) layout."
    },
    "Celup": {
        "description": "Also known as Tie-Dye or Jumputan. It is made by tying sections of fabric before dyeing, creating unique, abstract, and colorful patterns.",
        "pattern": "Abstract, radiating, or circular patterns created by binding the fabric, resulting in soft edges and color gradients."
    },
    "Cendrawasih": {
        "description": "A signature motif from Papua, featuring the Bird of Paradise (Cendrawasih). It symbolizes the beauty and richness of Eastern Indonesia's nature.",
        "pattern": "Features the distinctive Bird of Paradise with elaborate tail feathers, often set against a background of local flora or tribal geometric accents."
    },
    "Ceplok": {
        "description": "A geometric pattern based on squares, circles, or rosettes. It represents balance, order, and the harmonious structure of the universe.",
        "pattern": "Repeated geometric shapes (circles, squares, rosettes) arranged in a symmetric grid, often resembling flowers or coins."
    },
    "Ciamis": {
        "description": "Known as 'Batik Ciamisan', it features simple, naturalistic motifs with a distinct color palette, often lacking the heavy brown (sogan) of Central Java.",
        "pattern": "Simple, uncluttered naturalistic motifs (flora/fauna) on a light or white background, often using black, white, and yellowish-brown tones."
    },
    "Garutan": {
        "description": "From Garut, West Java. It is known for its pastel colors and simple, geometric or floral motifs, reflecting a calm and practical lifestyle.",
        "pattern": "Geometric arrangements (like diamond shapes) filled with small, simple floral or line patterns, often in soft pastel colors (gumading)."
    },
    "Gentongan": {
        "description": "A unique batik from Madura, colored in clay jars (gentong). It features bold colors and abstract designs, often symbolizing the sea and coastal life.",
        "pattern": "Bold, abstract, and expressive designs with strong colors (red, purple, green), often featuring sea creatures or plants."
    },
    "Kawung": {
        "description": "Inspired by the cross-section of an Areca palm fruit. It represents purity, self-control, and the hope that humans will be useful to society.",
        "pattern": "Four oval or elliptical shapes touching at the tips, arranged in a repetitive geometric grid, resembling a four-petaled flower or palm fruit cross-section."
    },
    "Keraton": {
        "description": "The royal batik of Yogyakarta and Surakarta. It uses earth tones (sogan) and follows strict rules, symbolizing wisdom, charisma, and nobility.",
        "pattern": "Classic, solemn patterns using Sogan colors (dark brown, yellowish-brown, white/black), often featuring specific forbidden motifs (larangan) like large Parang or Semen."
    },
    "Lasem": {
        "description": "Known as 'Batik Tiga Negeri' (Three Realms), it blends Javanese, Chinese (red color), and Dutch influences. It is famous for its intricate details.",
        "pattern": "Intricate blend of Chinese motifs (dragons, phoenixes, flowers) in bright red (getih pitik) with Javanese geometric backgrounds."
    },
    "Mega Mendung": {
        "description": "Originating from Cirebon, this cloud pattern symbolizes patience. Just as clouds hold rain to cool the earth, a leader must control their emotions.",
        "pattern": "Stylized cloud shapes with graduating layers of color (usually 5-7 layers) from dark to light, often in blue or red tones."
    },
    "Parang": {
        "description": "One of the oldest motifs, resembling waves breaking against a reef. It symbolizes continuous struggle, resilience, and the spirit of never giving up.",
        "pattern": "Diagonal parallel bands containing 'S' shaped knife-like patterns, interlocking continuously."
    },
    "Pekalongan": {
        "description": "From the 'City of Batik', featuring vibrant colors and naturalistic floral bouquets (buketan). It reflects the dynamic coastal culture and foreign influences.",
        "pattern": "Vibrant, multi-colored designs often featuring elaborate floral bouquets (buketan) and birds (hong), filling the space dynamically."
    },
    "Priangan": {
        "description": "A general term for batik from the Priangan region (Tasikmalaya, Ciamis, Garut). It is characterized by open patterns and a lighter, more natural feel.",
        "pattern": "Naturalistic flora and fauna motifs arranged openly (not densely packed), often on light backgrounds with soft colors."
    },
    "Sekar Jagad": {
        "description": "Literally meaning 'Flowers of the World.' It represents the diversity and beauty of different cultures coming together in harmony.",
        "pattern": "Map-like arrangement of irregular patches or islands, each filled with different contrasting patterns and motifs."
    },
    "Sidoluhur": {
        "description": "Means 'Hope for Nobility'. Traditionally worn by brides and grooms, it symbolizes the prayer for a noble and virtuous life together.",
        "pattern": "Geometric grid containing square or diamond frames filled with symbolic motifs like garuda wings, temples, or flowers."
    },
    "Sidomukti": {
        "description": "Means 'Hope for Happiness and Prosperity'. Worn during weddings, it symbolizes the wish for a happy, prosperous, and fulfilled future.",
        "pattern": "Similar to Sidoluhur, featuring geometric frames filled with motifs like butterflies or small shrines (meru), often on a bamboo-weave background."
    },
    "Sogan": {
        "description": "The classic brown batik of Central Java. The color comes from the Soga tree bark. It represents humility, down-to-earth nature, and tradition.",
        "pattern": "Characterized by its color palette of dark brown, yellowish-brown, and black/white. Can feature various classic motifs like Parang, Kawung, or Semen."
    },
    "Tambal": {
        "description": "Meaning 'Patchwork' or 'Mending'. It consists of various motif fragments. Traditionally believed to have healing properties for the sick.",
        "pattern": "Patchwork-like arrangement of triangles or squares, each containing a different batik motif (Parang, Ceplok, etc.)."
    },
    "Truntum": {
        "description": "Created by a Queen who felt neglected by the King. The star-like flowers symbolize unconditional love and a romance that blooms again in the darkness.",
        "pattern": "Small, repetitive star-like or jasmine flower motifs scattered evenly across a dark background, resembling a starry night sky."
    }
}

@app.get("/")
def read_root():
    return {"message": "Welcome to KenalBatik API"}

@app.post("/analyze")
async def analyze_batik(file: UploadFile = File(...)):
    # 1. READ IMAGE
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB (fixes issues with some PNGs having transparency)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 1b. SAVE TO TEMP
    file_id = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join("temp_uploads", file_id)
    image.save(temp_path)

    # 2. AI PREDICTION
    if model:
        # Prepare image for the brain
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            # Convert raw numbers to probabilities (0% - 100%)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get the winner
        confidence, class_idx = torch.max(probabilities, 0)
        predicted_class = CLASS_NAMES[class_idx.item()]
        conf_score = confidence.item()

        # --- Generate Heatmap ---
        cam = BatikGradCAM(model)
        # We need to use the same input tensor, but generate expects the tensor
        # It also needs the class index we want to visualize (the predicted one)
        heatmap_image = cam.generate(input_tensor, class_idx.item())
        
        # Convert PIL Image to Base64
        buffered = io.BytesIO()
        heatmap_image.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        # Emergency Fallback (If model failed to load)
        predicted_class = "Unknown"
        conf_score = 0.0
        heatmap_base64 = None

    # 3. GET METADATA
    info = BATIK_INFO.get(predicted_class, {
        "description": f"A beautiful Indonesian Batik pattern identified as {predicted_class}.",
        "pattern": "Pattern details not available."
    })

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": round(conf_score, 4),
        "description": info["description"],
        "pattern": info.get("pattern", "Pattern details not available."),
        "heatmap": heatmap_base64,
        "id": file_id
    }

class FeedbackRequest(BaseModel):
    file_id: str
    correct_label: str

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    # 1. Validate Label
    if feedback.correct_label not in CLASS_NAMES:
        return {"error": "Invalid Class Name"}

    # 2. Verify File Exists
    temp_path = os.path.join("temp_uploads", feedback.file_id)
    if not os.path.exists(temp_path):
        return {"error": "File not found or expired"}

    # 3. Move to Dataset
    target_dir = os.path.join("dataset", feedback.correct_label)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, feedback.file_id)
    shutil.move(temp_path, target_path)

    return {"message": "Image added to training data", "path": target_path}
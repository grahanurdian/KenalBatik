import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# 1. SETUP FOR M1 MAC (Metal Performance Shaders)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 2. DEFINE TRANSFORMATIONS
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20), # Rotate +/- 20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handle different lighting
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. LOAD DATA
data_dir = 'dataset' # Folder name
image_datasets = datasets.ImageFolder(data_dir, data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
class_names = image_datasets.classes
print(f"Classes found: {class_names}")

# 4. LOAD MODEL & MODIFY HEAD
model = models.mobilenet_v2(weights='DEFAULT')

# 1. Start by freezing everything
for param in model.features.parameters():
    param.requires_grad = False

# 2. UNFREEZE the last block of MobileNet (Layer 18)
# This allows the "high level" texture features to update
for param in model.features[-1].parameters():
    param.requires_grad = True

# 3. Replace the classifier as before
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

model = model.to(device)

# 4. LOWER the learning rate
# Since we are unfreezing layers, we must learn SLOWER to not break what it already knows
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Changed from 0.001 to 0.0001

# Freeze the "features" layers (we keep the pre-trained eyes)
for param in model.features.parameters():
    param.requires_grad = False

# Replace the "classifier" (the brain) with a new one for OUR 5 classes
# MobileNetV2's last layer is usually 1280 inputs -> 1000 outputs
# We change it to 1280 inputs -> 5 outputs
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

model = model.to(device)

# 5. TRAINING LOOP
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 30 # 30 runs through the dataset

print("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()  # Start epoch timer
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_end = time.time()  # End epoch timer
    epoch_time = epoch_end - epoch_start
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloaders):.4f} | Accuracy: {100 * correct / total:.2f}% | Time: {epoch_time:.2f}s")

# ... end of loop ...

end_time = time.time()  # <--- Snap the ending timestamp
elapsed_time = end_time - start_time

print(f"\nTraining complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

# 6. SAVE THE BRAIN
torch.save(model.state_dict(), "batik_model_v1.pth")
print("Model saved as batik_model_v1.pth")
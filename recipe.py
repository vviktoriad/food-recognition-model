import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import re
from pathlib import Path

class MiniNet(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = create_feature_extractor(
            resnet,
            return_nodes={'avgpool': 'features'}
        )
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)['features']
        x = features.flatten(1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f]

def load_model(model_path, num_classes):
    model = MiniNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def predict_image(model, img_path, class_names):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img)
    input_tensor = img_t.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()
    return class_names[pred]

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from pathlib import Path

# Ścieżka do folderu z modelem (TEN SAM FOLDER co na screenie)
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = os.path.join(SCRIPT_DIR, "recipe_model")  # Folder, nie plik .pt!

def generate_recipe(food_name: str) -> dict:
    try:
        # DEBUG: Wypisz ścieżkę, żeby sprawdzić czy dobra
        print(f"Ładuję model z: {MODEL_PATH}")
        
        # Tokenizer i model ŁADUJEMY Z FOLDERU
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        
        # Reszta Twojego kodu pozostaje bez zmian...
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        input_text = f"generate recipe: {food_name}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        output = model.generate(
            input_ids,
            max_length=512,
            num_beams=8,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parsowanie wyniku
        parts = decoded_output.split("Instrukcje:")
        ingredients = parts[0].replace("Składniki:", "").strip().split("\n")
        ingredients = [i.strip() for i in ingredients if i.strip()]
        
        instructions = parts[1].strip() if len(parts) > 1 else "Brak instrukcji."
        
        return {"Składniki": ingredients, "Instrukcje": instructions}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"Składniki": [], "Instrukcje": "Coś się zepsuło, model nie działa."}

if __name__ == "__main__":
    food_name = 'spaghetti'
    recipe = generate_recipe(food_name)
    print(f"Przepis dla: {food_name}")
    print("Składniki:")
    for ing in recipe["Składniki"]:
        print(f"- {ing}")
    print("\nInstrukcje:")
    print(recipe["Instrukcje"])
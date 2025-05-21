import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

# Część 1: Model rozpoznawania jedzenia ze zdjęcia
class FoodClassifier(nn.Module):
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

def load_classifier_model(model_path, num_classes):
    model = FoodClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Część 2: Model generowania przepisów
def load_recipe_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, add_prefix_space=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model

def generate_recipe(food_name, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_text = f"generate recipe: {food_name}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    try:
        output = model.generate(
            input_ids,
            max_length=200,  # Reduced max length
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        # print(f"Raw model output: {decoded_output}")  # Debugging
        
        # Parsing the result
        if "Instrukcje:" in decoded_output:
            parts = decoded_output.split("Instrukcje:")
            ingredients_part = parts[0].replace("Skadniki:", "").strip()
            
            # Extract ingredients
            if "[" in ingredients_part and "]" in ingredients_part:
                # Parse the list format correctly
                ingredients = ingredients_part.strip("[]").replace("'", "").split(", ")
            else:
                # Split by spaces and join with commas if no list format is detected
                ingredients = [i.strip() for i in ingredients_part.split(" ") if i.strip()]
                ingredients = [", ".join(ingredients)]  # Join ingredients with commas
            
            instructions = parts[1].strip() if len(parts) > 1 else "No instructions found."
        else:
            ingredients = []
            instructions = "Failed to find instructions in the model output."
        
        return {"Ingredients": ingredients, "Instructions": instructions}
    except Exception as e:
        print(f"Error while generating recipe: {str(e)}")
        return {"Ingredients": [], "Instructions": "Failed to generate recipe."}

# Główna funkcja aplikacji
def main():
    # Ścieżki do modeli i danych
    classifier_model_path = os.path.join("model", "food_matching_model.pt")
    classes_file = os.path.join("model", "classes.txt")
    recipe_model_dir = os.path.join("recipe_model")
    input_dir = "input"
    
    # Sprawdzenie czy istnieje dokładnie jeden plik w folderze input
    allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(allowed_ext)]
    
    if len(files) != 1:
        print("The 'input' folder must contain exactly one image file!")
        return
    
    img_path = os.path.join(input_dir, files[0])
    
    # Transformacje dla obrazu
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Załadowanie modelu klasyfikującego
    class_names = load_classes(classes_file)
    classifier_model = load_classifier_model(classifier_model_path, num_classes=len(class_names))
    
    # Rozpoznanie dania ze zdjęcia
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img)
    input_tensor = img_t.unsqueeze(0)
    
    with torch.no_grad():
        output = classifier_model(input_tensor)
        pred = output.argmax(1).item()
    
    food_name = class_names[pred]
    print(f"\nRecognized dish: {food_name}")
    
    # Load the recipe generation model
    try:
        tokenizer, recipe_model = load_recipe_model(recipe_model_dir)
        recipe = generate_recipe(food_name, tokenizer, recipe_model)
        
        # Display the recipe
        print("\nRECIPE:")
        print("Ingredients:")
        for ingredient in recipe["Ingredients"]:
            print(f"- {ingredient}")
        print("\nInstructions:")
        print(recipe["Instructions"])
    except Exception as e:
        print(f"\nFailed to generate recipe: {str(e)}")

if __name__ == "__main__":
    main()
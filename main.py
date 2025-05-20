import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

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

if __name__ == "__main__":
    model_path = os.path.join("model", "food_matching_model.pt")
    classes_file = os.path.join("model", "classes.txt")
    input_dir = "input"

    allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(allowed_ext)]
    if len(files) != 1:
        print("W folderze 'input' musi być dokładnie jeden plik graficzny!")
        exit(1)
    img_path = os.path.join(input_dir, files[0])

    class_names = load_classes(classes_file)
    model = load_model(model_path, num_classes=len(class_names))

    predicted = predict_image(model, img_path, class_names)
    print(f"Model rozpoznał to jedzenie jako: {predicted}")
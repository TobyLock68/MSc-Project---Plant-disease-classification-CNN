import torch
import random
import numpy as np
import os
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset_dictionary import PLANTDOC_TO_PLANTVILLAGE
from sklearn.metrics import precision_recall_fscore_support

def seed_reproducibility(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_reproducibility(42)

#same starting code as training with slight name change
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

test_transformation = transforms.Compose([
    transforms.Resize((224,224)),    #makes images all standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])     #ImageNet dataset exact averages
])

#directories
root = Path(__file__).parent.parent
plantvil_dir = root/"data" / "plantvillage dataset" / "color"
plantdoc_dir = root/"data" / "PlantDoc-Dataset"

#data loading
plantdoc_data = ImageFolder(root=str(plantdoc_dir), transform=test_transformation)
test_loader = DataLoader(plantdoc_data, batch_size = 32, shuffle = False)

#index the dictionary
pv_classes_sort = sorted([d for d in os.listdir(plantvil_dir) if os.path.isdir(plantvil_dir / d) and not d.startswith('.')])
class_to_idx = {name: idx for idx, name in enumerate(pv_classes_sort)}

pd_to_pv_idx = {}

for pd_idx, pd_name in enumerate(plantdoc_data.classes):
    if pd_name in PLANTDOC_TO_PLANTVILLAGE:
        matched = PLANTDOC_TO_PLANTVILLAGE[pd_name]
        if matched in class_to_idx:
            pd_to_pv_idx[pd_idx] = class_to_idx[matched]
        else:
            pd_to_pv_idx[pd_idx] = -1
    else:
        pd_to_pv_idx[pd_idx] = -1


#function to load and test model weights
def model_eval(weight_file_path):
    #shortened model setup as we are passing weights to the model from training
    model = models.resnet50(weights = None)
    model.fc = torch.nn.Linear(model.fc.in_features, 38)

    model.load_state_dict(torch.load(f"../models/{weight_file_path}", map_location=device))
    model = model.to(device)
    model.eval()

    #setting up of variables to track metrics
    all_preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:          #begins loop pf batched images
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)         #forward pass
            _, preds = torch.max(outputs, 1)        #disctates the class/prediction

            mapped_labels = torch.tensor([pd_to_pv_idx[l.item()] for l in labels]).to(device)
            valid = mapped_labels != -1
            
            if valid.sum() > 0:
                all_preds.extend(preds[valid].cpu().numpy())
                targets.extend(mapped_labels[valid].cpu().numpy())
            
    #convert arrays to Numpy format
    all_preds = np.array(all_preds)
    targets = np.array(targets)
    total = len(targets)

    #accuracy
    correct = (all_preds == targets).sum()
    accuracy = (correct/total)*100 if total > 0 else 0.0

    #recall, precision and F1-score using Scikit leanr

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        targets, all_preds, average='macro', zero_division=0
    )

    print(f"Accuracy for {weight_file_path} on PlantDoc test = {accuracy:.2f}%")
    print(f"Recall for {weight_file_path} on PlantDoc test = {recall:.2f}%")
    print(f"Precision for {weight_file_path} on PlantDoc test = {precision:.2f}%")
    print(f"F1-score for {weight_file_path} on PlantDoc test = {f1_score:.2f}%")


    return accuracy

#run baseline and augemented in parallel
baseline_test_acc = model_eval("baseline_training_accuracies.pth")
augmented_test_acc = model_eval("augmented_training_accuracies.pth")
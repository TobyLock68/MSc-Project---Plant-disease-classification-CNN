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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

#same starting code as training with slight name change

def seed_reproducibility(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_reproducibility(42)

# ---- COLAB SETUP ---
root_dir = Path("/content")
plantdoc_dir = root_dir / "PlantDoc-Dataset"
plantvil_dir = root_dir/"plantvillage dataset"/ "color"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
num_workers_cfg = 2

# --- LOCAL SETUP ---
# root = Path(__file__).parent.parent
# plantdoc_dir = root/"data" / "PlantDoc-Dataset"
# plantvil_dir = root_dir/"data"/"plantvillage dataset"/ "color"
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# num_workers_cfg = 0

model_save_dir = root_dir/"models"
model_save_dir.mkdir(exist_ok=True)

#best fold for each training run determined by average fold accuracy
baseline_best_fold = 5
augemented_best_fold = 2

# --- Data setup ---

test_transformation = transforms.Compose([
    transforms.Resize((224,224)),    #makes images all standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])     #ImageNet dataset exact averages
])

#data loading
plantdoc_data = ImageFolder(root=str(plantdoc_dir), transform=test_transformation)
test_loader = DataLoader(plantdoc_data, batch_size = 32, shuffle = False, num_workers=num_workers_cfg)

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
def model_eval(experiment, best_fold_idx):
    
    #Runs and ensemble method using average training weights from each fold, also runs a best fold testing
    
    #shortened model setup as we are passing weights to the model from training
    model_list = []

    for i in range(1,6):
        file_name = f"{experiment}_fold_{i}.pth"
        checkpoint_path = model_save_dir / file_name

        if not checkpoint_path.exists():
            raise FileNotFoundError("Couldn't load weights")
        
        model = models.resnet50(weights = None)
        model.fc = torch.nn.Linear(model.fc.in_features, 38)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        model_list.append(model)

# -- Best Fold Isolated Approach --

    #identify best fold
    best_model = model_list[best_fold_idx - 1]

    #metric calculation tracking variables
    best_fold_preds = []
    best_fold_probs = []
    ensemble_preds = []
    ensemble_probs = []
    targets = []

    individual_fold_preds = [[] for _ in range(5)]

    with torch.no_grad():
        for inputs, labels in test_loader:          #begins loop pf batched images
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)

            mapped_labels = torch.tensor([pd_to_pv_idx[l.item()] for l in labels]).to(device)
            valid = mapped_labels != -1

            if valid.sum() > 0:
                best_outputs = best_model(inputs)
                _, b_preds = torch.max(best_outputs, 1)
                b_probs = torch.softmax(best_outputs, 1)

                # --- Ensemble approach ----

                accum_probs = torch.zeros((batch_size, 38), device=device)

                for idx, model in enumerate(model_list):
                    outputs = model(inputs)
                    _, fold_pred = torch.max(outputs, 1)
                    individual_fold_preds[idx].extend(fold_pred[valid].cpu().numpy())

                    accum_probs += torch.softmax(outputs, dim=1)

                _, ens_preds = torch.max(accum_probs, 1)
                ens_probs = accum_probs/ len(model_list)

                #store the matching classification parts
                best_fold_preds.extend(b_preds[valid].cpu().numpy())
                ensemble_preds.extend(ens_preds[valid].cpu().numpy())
                targets.extend(mapped_labels[valid].cpu().numpy())
                best_fold_probs.extend(b_probs[valid].cpu().numpy())
                ensemble_probs.extend(ens_probs[valid].cpu().numpy())

    
    #convert to numpy strucrture
    best_fold_preds = np.array(best_fold_preds)
    ensemble_preds = np.array(ensemble_preds)
    targets = np.array(targets)
    total = len(targets)
    best_fold_probs = np.array(best_fold_probs)
    ensemble_probs = np.array(ensemble_probs)

    #saving raw data for later AUC graph 
    save_path = model_save_dir / f"{experiment}_eval_data.npz"

    np.savez(
        save_path,
        targets=targets,
        best_fold_preds=best_fold_preds,
        best_fold_probs=best_fold_probs,
        ensemble_preds=ensemble_preds,
        ensemble_probs=ensemble_probs
    )

    print(f"RAW DATA SAVE TO: {save_path}")

    #function for confusion matrix
    def confusion_matrix_print(preds, label_text):
        unique_classes = np.unique(targets)
        current_classes = [pv_classes_sort[idx] for idx in unique_classes]

        con_mat = confusion_matrix(targets, preds, labels = unique_classes)
        print(f"\n CONFUSION MATRIX BREAKDOWN ({label_text})")
        print(f"{'Class Name':<40} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'TN':<5}")

        for i, name in enumerate(current_classes):
            tp = con_mat[i, i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = np.sum(con_mat) - (tp + fp + fn)

            print(f"{name:<40} | {tp:<5} | {fp:<5} | {fn:<5} | {tn:<5}")

    #funtion for performance metrics

    def metric_summary(preds, label_text):
        correct = (preds == targets).sum()
        accuracy = (correct/total)*100 if total > 0 else 0.0

    #recall, precision and F1-score using Scikit leanr

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )

        print(f"Accuracy for {label_text} on PlantDoc test = {accuracy:.2f}%")
        print(f"Recall for {label_text} on PlantDoc test = {recall:.2f}%")
        print(f"Precision for {label_text} on PlantDoc test = {precision:.2f}%")
        print(f"F1-score for {label_text} on PlantDoc test = {f1_score:.2f}%")

        return accuracy

    print("Metric breakdown for BEST SINGLE FOLD:")
    best_acc = metric_summary(best_fold_preds, f"{experiment}_fold_{best_fold_idx}.pth")
    confusion_matrix_print(best_fold_preds, f"{experiment}_fold_{best_fold_idx}.pth")

    print("Metric breakdown for ENSEMBLE METHOD:")
    ensemble_acc = metric_summary(ensemble_preds, f"{experiment}_ensemble")
    confusion_matrix_print(ensemble_preds, f"{experiment}_ensemble")

    #print section for individual folds in testing
    print(f"Individual fold accuracies ({experiment}):")
    for idx in range(5):
        f_preds = np.array(individual_fold_preds[idx])
        correct = (f_preds == targets).sum()
        acc = (correct/total)*100 if total > 0 else 0.0
        print(f"Fold: {idx + 1} -- Accuracy: {acc:.4f}%")

    return best_acc, ensemble_acc

#run baseline and augemented in parallel
baseline_best, baseline_ensemble = model_eval("resnet50_baseline", baseline_best_fold)
augmented_best, augemented_ensemble = model_eval("resnet50_augmented", augemented_best_fold)
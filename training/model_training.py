import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def seed_reproducibility(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_reproducibility(42)

#directory setup for google colab
root_dir = Path("/content")
experiment_name = "baseline"
plantvil_dir = root_dir/"plantvillage dataset"/ "color"

experiment_name = "augmented"
plantvil_dir = root_dir/"plantvillage_augmented"


#directory and hardware setup for local

#root_dir = Path(__file__).parent.parent
#different directories for augmented and baseline data
#experiment_name = "baseline"
#plantvil_dir = root_dir/"data"/"plantvillage dataset"/ "color"
#experiment_name = "augmented"
#plantvil_dir = root_dir/"data"/"plantvillage_augmented"


model_save_dir = root_dir/"models"
model_save_dir.mkdir(exist_ok=True)

#checks if better gpu hardware available (may or may not work on my mac)
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#google colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


#data/ image preparation:
img_transformation = transforms.Compose([
    transforms.Resize((224,224)),    #makes images all standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])     #ImageNet dataset exact averages
])

if __name__ == "__main__":
    #data loading
    data = datasets.ImageFolder(root = str(plantvil_dir), transform = img_transformation)
    num_labels = [sample[1] for sample in data.samples]

    #cross val 
    #splitting logic
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
        #specifies no. splits, randomisation of data (we are in alphabetical order) and set seed


    fold_accuracies = []    #stores accuracies to given baseline at end

    #looping cross val
    fold_splits = skf.split(X=np.zeros(len(num_labels)), y = num_labels)

    for fold, (train_idx, val_idx) in enumerate(fold_splits, start = 1):
        print(f"K-Fold Progress: Fold {fold}/5 ({experiment_name})")

        #creates the data subsets
        train_subset = Subset(data, train_idx)
        validation_subset = Subset(data, val_idx)

        #preps data subsets to be fed into model
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        validation_loader = DataLoader(validation_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        #model setup
        model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

        #freezing of convolutional layers 
        for param in model.parameters():
            param.requires_grad = False

        #global average pooling 
        in_features = model.fc.in_features

        #replacement of 1000-output layer to our max of 38 classes and apply softmax
        model.fc = nn.Linear(in_features, 38)

        model = model.to(device)    #sends to previously defined GPU if available

        optimiser = optim.Adam(model.fc.parameters(), lr=0.001)     #targeted training and learning algorithm
        criteria = nn.CrossEntropyLoss()    #loss function (how wrong models guess was)

        #--------- ACTTUAL TRAINING CODE -------------

        best_validation_accuracy = 0.0
        best_weights = copy.deepcopy(model.state_dict())

        for epoch in range (1,16):      #15 training epochs for each fold as per literature
            model.train()

            for batch_idx, (inputs, batch_labels) in enumerate(train_loader, start=1):
                inputs, batch_labels = inputs.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)

                optimiser.zero_grad()       #removes previous gradients
                outputs = model(inputs)     #forward pass
                loss =criteria(outputs, batch_labels)
                loss.backward()     #backward pass
                optimiser.step()        #update weights

                if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                    print(f" EPOCH = {epoch}/15 --- Batch {batch_idx}/{len(train_loader)}")

                #validation stage within the training loop

            model.eval()
            validation_correct = 0
            with torch.no_grad():
                for inputs, batch_labels in validation_loader:
                    inputs, batch_labels = inputs.to(device), batch_labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    validation_correct += torch.sum(preds ==  batch_labels.data)
            
            epoch_accuracy = (validation_correct.double() / len(val_idx)).item()

            print(f" Epoch {epoch}/15 | Accuracy = {epoch_accuracy:.4f}")

            #tracking the best weights for each fold
            if epoch_accuracy > best_validation_accuracy:
                best_validation_accuracy = epoch_accuracy
                best_weights = copy.deepcopy(model.state_dict())

        fold_accuracies.append(best_validation_accuracy)

        model_filename = f"resnet50_{experiment_name}_fold_{fold}.pth"
        torch.save(best_weights, model_save_dir / model_filename)
        print(f"Optimal weights for fold saved: models/{model_filename}")


    print(f"Training finished. Average accuracy = {np.mean(fold_accuracies):.4f}\n")
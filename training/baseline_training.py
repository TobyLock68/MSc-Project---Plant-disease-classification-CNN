import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold

#directory and hardware setup
root_dir = Path(__file__).parent.parent
plantvil_dir = root_dir/"data"/"plantvillage dataset"/ "color"
model_save_dir = root_dir/"models"
model_save_dir.mkdir(exist_ok=True)

#checks if better gpu hardware available (scalable not useful for me)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#data/ image preparation:
img_transformation = transforms.Compose([
    transforms.Resize((224,224)),    #makes images all standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])     #ImageNet dataset exact averages
])

#data loading
data = datasets.ImageFolder(root = str(plantvil_dir), transform = img_transformation)
num_labels = [sample[1] for sample in data.samples]

#cross val 
#splitting logic
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    #specifies no. splits, randomisation of data (we are in alphabetical order) and set seed


fold_accuracies = []    #stores accuracies to given baseline at end


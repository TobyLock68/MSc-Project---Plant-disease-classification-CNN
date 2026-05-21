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

#data preparation:

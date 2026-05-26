import os
from pathlib import Path
from dataset_dictionary import PLANTDOC_TO_PLANTVILLAGE

ROOT_DIR = Path(__file__).parent.parent
PV_DIR = ROOT_DIR / "data" / "plantvillage dataset" / "color"
PD_DIR = ROOT_DIR / "data" / "PlantDoc-Dataset" / "combined_test"

def check_datasets():
    pv_folders = set(d for d in os.listdir(PV_DIR) if os.path.isdir(PV_DIR / d) and not d.startswith('.'))
    pd_folders = set(d for d in os.listdir(PD_DIR) if os.path.isdir(PD_DIR / d) and not d.startswith('.'))
    
    # intersecting classes based on dictionary
    intersects = {k: v for k, v in PLANTDOC_TO_PLANTVILLAGE.items() if k in pd_folders and v in pv_folders}
    
    print(f"PlantVillage physical classes: {len(pv_folders)}")
    print(f"PlantDoc physical classes:     {len(pd_folders)}")
    print(f"Exact matching classes found:   {len(intersects)}")
    print("-" * 70)
    print("LIST OF INTERSECTING CLASSES:")
    
    for idx, (pd_name, pv_name) in enumerate(sorted(intersects.items()), 1):
        print(f" {idx:2d}. PlantDoc: '{pd_name}' ── PlantVillage: '{pv_name}'")
        
    print("=" * 70)

if __name__ == "__main__":
    if not PV_DIR.exists() or not PD_DIR.exists():
        print(f"❌ Path Error: Ensure your folders exist at:\n   {PV_DIR}\n   {PD_DIR}")
    else:
        check_datasets()
 # MSc Project: Enhancing CNN robustness using an image distorting pipeline to create synthetically distorted dataset

A computer vision pipeline designed to classify plant diseases across 38 distinct categories using a ResNet50 architecture. This project systematically investigates model robustness and domain generalization by comparing a baseline performance framework against a dual-layer stochastic environmental distortion training pipeline.

---

## Project Overview & Methodology

When deploying computer vision models for agricultural applications in the real world, models trained on sterile, lab-curated images (like the PlantVillage dataset) often fail due to unpredictable field environments. 

This project develops a structured, multi-layered data synthesis and evaluation pipeline designed to quantify, test, and analyze the impact of complex environmental field conditions on deep convolutional neural networks.

1. **The Baseline:** A pre-trained ResNet50 model optimized entirely on pristine, unaltered plant leaf images.
2. **The Augmented Defense:** An identical ResNet50 model exposed to a dual-layered stochastic distortion pipeline designed to simulate realistic sensor and atmospheric degradation:
   * **Layer 1 (Sensor/Camera Realism):** Dynamic motion blur, non-linear HLS color-space lighting adjustments, and random noise injections (OpenCV/NumPy).
   * **Layer 2 (Atmospheric Realism):** Random physical phenomena simulations like heavy fog overlay and camera lens solar flare artifacts (Albumentations).

---

## Project Structure

```text
MSc-Project---Plant-disease-classification-CNN/
├── coded_parts/
│   ├── distortion_layer_1.py       # Layer 1: Custom OpenCV functions (Motion blur, Light intensity, Noise)
│   └── distortion_layer_2.py       # Layer 2: Albumentations functions (Fog, Sun Flare)
├── data/
│   ├── PlantDoc-Dataset/           # Real-world, out-of-distribution verification set
│   ├── plantvillage dataset/       # Baseline PlantVillage dataset (Clean)
│   ├── plantvillage_augmented/     # Target location for distorted dataset
│   ├── test_plantvillage/          # Isolated clean test split
│   ├── test_plantvillage_augmented/# Isolated distorted test split
│   ├── PlantDoc.zip                # Compressed raw PlantDoc data
│   ├── plantvillage_augmented.zip  # Compressed augmented dataset
│   └── PlantVillage.zip            # Compressed baseline dataset
├── models/                         # Local storage for trained .pth checkpoints
├── my_custom_distortions/          # Folder for results of image_visual.py
├── training/
│   ├── dataset_dictionary.py       # Dictionary of class labels used for the alignment check 
│   ├── model_testing.py            # Core testing scritp architecture
│   ├── model_training.py           # Core training script architecture
│   └── verify_alignment.py         # Check ensuring label alignment between datasets
├── .gitignore                      # Prevents committing venv caches and large datasets
├── image_visual.py                 # Provides a visual insight into each distortions
├── README.md                       # Project overview doc
├── requirements.txt                # Project dependencies
└── run-pipeline.py                 # Script to construct the augmented dataset
```



### Clone the repository

 ```bash
git clone [https://github.com/tobylock/MSc-Project---Plant-disease-classification-CNN.git](https://github.com/tobylock/MSc-Project---Plant-disease-classification-CNN.git)
```
### Navigate to the root of the directory

```bash
cd MSc-Project---Plant-disease-classification-CNN
```
### Create the virtual environment

```bash
python3 -m venv venv   
```
### Activate the virtual environment

```bash
source venv/bin/activate
```

### Upgrade PIP and install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The scripts are to be run sequentially from the project root directory in order to parse, visualise distortions and construct the augmented dataset.

Before running an augmentations ot training we need to verify the class labels in the two datasets are aligned/ match. This prevents class labelling errors later down the line and ensure that in training a testing phases the model is generalising correctly.

```bash
python3 training/verify_alignment.py
```
Now we can see can run the following code to get an idea what each distortion will do to the original image. This is an optional section but is included for the project write up so that the report can be read and the distortions visualised.

```bash
python3 image_visual.py
```
Check the **my_custom_distortions/** folder to view the output examples

Now we can run the main distortion script. This script will create a dataset the same size as the original PlantVillage dataset. It iterates its way through the images and uses the functions found inside the **coded_parts/** file to distort the images storing them in final **data/plantvillage_augmented/** file:

```bash
python3 run_pipeline.py
```
## Model training

The training was originally written to run on my local Mac but due to the 5-fold cross-validation loop on thousands of images the training is very computationally intensive. As a result, this file also includes the necessary code adaptations to run on the cloud -- in this case Google Colab.

Using Google Colab we can utilise the faster T4 GPU or the A100 GPU dependent on packages available.

If your local machine has the capabilities to run this intensive code simply un-comment the areas labelled for local use and run the following command in the terminal:

```bash
python3 model_training.py
```
To train the models in th cloud follow this walkthrough:

#### Step 1: Upload the datasets
Compress the original unaltered PlantVillage dataset and the augmented target dataset into a **.zip** file and upload it directly into the Google Drive storage area (inside a folder named MSc_Project/)

#### Step 2: Setup Google colab notebook
Open a new Google Colab notebook and change the runtime option to the T4 GPU or the A100 GPU if available by navigating to 

```text
Runtime ➔ Change runtime type ➔ Hardware Accelerator ➔ T4
```

#### Step 3: Mount your Google drive and copy the start code
Paste and execute the following code block into Google Colab. This allows Colab to connect to your personal cloud/ Drive files and unzip the dataset ready for use.

```bash
# 1. Mount Google Drive (pop up will ask for permission)
from google.colab import drive
drive.mount('/content/drive')

# 2. Unzip your dataset from your Drive folder into Colab's local space
!unzip -q "/content/drive/MyDrive/MSc_Project/plantvillage_augmented.zip" -d "/content/"
print("Augmented Dataset unzipped!")
```

#### Step 4: Paste and execute the training code
Copy the entire training script into the Colab notebook ensuring the hardware targets the NVIDIA hardware acceleration. All the code is present under commented sections labelled Colab but should look similar to the beloew:

```bash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Hardware Accelerator: {device}")
```
It is important to check the correct pathways are setup for the dataset to be trained: 
e.g.

```bash
root_dir = Path("/content")
experiment_name = "augmented"  # Change to "baseline" or "augmented" depending on your run
plantvil_dir = root_dir / "plantvillage_augmented" #change dataset name depending on run
```

Running the cell will result in the 5-fold cross validation training to execute and saves new **.pth** files containing weights for each independent fold

## Model testing

Once the training has been completed download and drop the saved **.pth** files into the local **models/** folder on VS_code. From here we can run the **model_testing.py** file which tests the two trained models on the unseen PlantDoc dataset. 

For testing the code runs two methods of testing: 
1) We take the best fold from the training and use the weights from this fold in the model for testing 
2) We create and ensemble method which averages the weights from all five folds and uses these calculated weights for the model testing

The code once again can be run on a local machine but was altered to run on Google Colab to significantly reduce runtime. 

If running local simply copy and execute the following command in the terminal:

```bash
python3 training/model_testing.py
```

If running on Google Colab follow the below step for setup.

**-- ADD STEPS FOR COLAB SETUP**

## Resulting performance metrics
After testing is complete you should see a variety of performance metrics printed in the terminal. These metrics will include Accuracy, Precision, Recall, F1-score and confusion matrices and will be produced for both the baseline and augmented models. Again, we should see a print out for the best fold AND the ensemble method.

Example terminal:


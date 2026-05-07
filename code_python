# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

ultrasound_nerve_segmentation_path = kagglehub.competition_download('ultrasound-nerve-segmentation')

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 1. Force images to show inside the notebook
%matplotlib inline

class KaggleNerveDataset(Dataset):
    def __init__(self, image_folder, img_size=(256, 256)):
        self.img_size = img_size
        
        # Check if folder exists
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Folder not found: {image_folder}. Please check your Kaggle input path.")
        
        # Get all image files (excluding mask files)
        all_files = sorted(os.listdir(image_folder))
        self.image_paths = [os.path.join(image_folder, f) for f in all_files if "_mask" not in f]
        
        # Build mask paths by replacing extension with _mask.tif
        self.mask_paths = [f.replace(".tif", "_mask.tif") for f in self.image_paths]
        
        print(f"✅ Found {len(self.image_paths)} images in {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read as Grayscale
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Handle cases where image might be missing
        if img is None or mask is None:
            print(f"Warning: Could not read file at index {idx}")
            return torch.zeros((1, *self.img_size)), torch.zeros((1, *self.img_size))

        # Resize and Normalize
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        # Convert to Tensors (Channels, Height, Width)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        return img_tensor, mask_tensor

# Visualization Hook 
def visualize_input(img, mask):
    # Convert back to numpy and remove channel for plotting
    img_np = img.squeeze().numpy()
    mask_np = mask.squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    
    # Raw Ultrasound Plot
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Raw Ultrasound")
    plt.axis('off')
    
    # Mask Plot (using a color map to make it pop)
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='viridis') 
    plt.title("Ground Truth Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# RUNNING THE CODE 
# Update 'train_data_path' to wherever your Kaggle data is extracted
train_data_path = '/kaggle/input/ultrasound-nerve-segmentation/train/' 

try:
    # 1. Initialize Dataset
    nerve_data = KaggleNerveDataset(image_folder=train_data_path)

    # 2. Extract one sample
    sample_img, sample_mask = nerve_data[0]

    # 3. Show output
    visualize_input(sample_img, sample_mask)
except Exception as e:
    print(f"Error: {e}")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

%matplotlib inline

class DiffusionDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple Residual CNN for cleaning
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        # Predicts noise and subtracts it (Residual Learning)
        noise_map = self.conv(x)
        return torch.clamp(x - noise_map, 0, 1)

# --- Rectified Visualization Hook ---
def visualize_denoising(raw, denoised):
    # Move to CPU and remove batch dimension for plotting
    raw_img = raw[0].detach().cpu().squeeze().numpy()
    denoised_img = denoised[0].detach().cpu().squeeze().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img, cmap='gray')
    plt.title("Original (Noisy)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img, cmap='gray')
    plt.title("Denoised (Diffusion)")
    plt.axis('off')
    
    plt.show()

#NEW: EXECUTION CODE
# 1. Instantiate the model
model = DiffusionDenoiser()

# 2. Prepare a sample (Use a sample from your nerve_data if loaded)
# If nerve_data isn't loaded yet, we'll create a dummy noisy tensor to test:
try:
    # Use real data if available from Part 1
    raw_sample, _ = nerve_data[0] 
    raw_sample = raw_sample.unsqueeze(0) # Add batch dimension (1, 1, 256, 256)
except NameError:
    # Fallback: Create a noisy synthetic image for testing
    print("Dataset not found, using synthetic test image...")
    raw_sample = torch.rand((1, 1, 256, 256)) 

model.eval() # Set to evaluation mode
with torch.no_grad():
    denoised_output = model(raw_sample)

visualize_denoising(raw_sample, denoised_output)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

%matplotlib inline

class EdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels to find horizontal (kx) and vertical (ky) gradients
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1,1,3,3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        # Apply convolution with fixed Sobel filters
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        # Calculate magnitude of gradient
        edge_map = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return torch.clamp(edge_map, 0, 1)

#  Rectified Visualization Hook 
def visualize_edges(edge_map_tensor):
    # Convert tensor to numpy for plotting
    # .detach() removes gradient info, .cpu() moves to processor, .squeeze() removes extra dims
    edge_np = edge_map_tensor.detach().cpu().squeeze().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(edge_np, cmap='magma')
    plt.title("Edge-Aware Map (Nerve Boundaries)")
    plt.colorbar(label='Edge Intensity')
    plt.axis('off')
    plt.show()

#EXECUTION BLOCK 
try:
    # 1. Initialize the Edge Extractor
    extractor = EdgeExtractor()

    # 2. Get the input (Ideally the 'denoised_output' from Part 2)
    # If Part 2 was run, use its output. Otherwise, we use a sample from dataset.
    if 'denoised_output' in locals():
        edge_input = denoised_output
    else:
        # Fallback: get a sample from dataset
        sample_img, _ = nerve_data[0]
        edge_input = sample_img.unsqueeze(0) 

    # 3. Run the forward pass
    extractor.eval()
    with torch.no_grad():
        extracted_edges = extractor(edge_input)

    # 4. TRIGGER THE OUTPUT
    visualize_edges(extracted_edges)

except NameError:
    print("Error: Dataset or Denoised Output not found. Please run the previous cells first!")
except Exception as e:
    print(f"An error occurred: {e}")

!pip install -q segmentation-models-pytorch

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
plt.rcParams.update({'font.size': 18})

%matplotlib inline

class EdgeFusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = DiffusionDenoiser()
        self.edge_extractor = EdgeExtractor()
        
        # We use a 3-channel UNet to process the fused information
        # in_channels=3 because we stack [Raw, Denoised, Edges]
        self.segmentor = smp.Unet(
            encoder_name="resnet18", 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=1
        )

    def forward(self, x):
        # Step 1: Denoise
        x_denoised = self.denoiser(x)
        # Step 2: Extract Edges from denoised version
        x_edges = self.edge_extractor(x_denoised)
        # Step 3: Fusion (Stacking into 3 channels along the 'Channel' dimension)
        fused = torch.cat([x, x_denoised, x_edges], dim=1)
        # Step 4: Final Mask
        mask = self.segmentor(fused)
        return torch.sigmoid(mask), x_denoised, x_edges

# --- Rectified Final Prediction Visualization ---
def visualize_results(raw, denoised, edges, pred, gt):
    # Helper to convert tensor to numpy
    def to_np(t):
        return t.detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    
    axes[0].imshow(to_np(raw), cmap='gray'); axes[0].set_title("1. Raw Input")
    axes[1].imshow(to_np(denoised), cmap='gray'); axes[1].set_title("2. Denoised")
    axes[2].imshow(to_np(edges), cmap='magma'); axes[2].set_title("3. Edges")
    axes[3].imshow(to_np(pred), cmap='jet'); axes[3].set_title("4. Prediction")
    axes[4].imshow(to_np(gt), cmap='gray'); axes[4].set_title("5. Target (GT)")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

#EXECUTION BLOCK 
try:
    # 1. Initialize the combined model
    full_model = EdgeFusionUNet()
    full_model.eval()

    # 2. Get a real sample from the dataset
    # Make sure 'nerve_data' was initialized in Part 1
    input_img, target_mask = nerve_data[0]
    input_batch = input_img.unsqueeze(0) # Add batch dimension

    # 3. Run the full pipeline
    with torch.no_grad():
        pred_mask, denoised_img, edge_map = full_model(input_batch)

    # 4. TRIGGER THE VISUALIZATION
    visualize_results(input_batch, denoised_img, edge_map, pred_mask, target_mask)

except Exception as e:
    print(f"Error: {e}")
    print("Hint: Make sure Part 1, 2, and 3 are run before this cell.")



!pip install -q albumentations
!pip install -q timm
!pip install -q transformers
!pip install -q seaborn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    jaccard_score,
    confusion_matrix,
    roc_curve,
    auc
)

import seaborn as sns
import pandas as pd
import time


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# DATA AUGMENTATION

train_transform = A.Compose([

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.2),

    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=15,
        p=0.5
    ),

    A.Normalize(mean=(0.5,), std=(0.5,)),

    ToTensorV2()

])

val_transform = A.Compose([

    A.Normalize(mean=(0.5,), std=(0.5,)),

    ToTensorV2()

])


# UPDATED DATASET CLASS
class KaggleNerveDataset(Dataset):

    def __init__(self,
                 image_folder,
                 img_size=(256,256),
                 transform=None):

        self.img_size = img_size
        self.transform = transform

        all_files = sorted(os.listdir(image_folder))

        self.image_paths = [

            os.path.join(image_folder, f)

            for f in all_files

            if "_mask" not in f

        ]

        self.mask_paths = [

            f.replace(".tif", "_mask.tif")

            for f in self.image_paths

        ]

        print(f"Loaded {len(self.image_paths)} images")

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(
            self.image_paths[idx],
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            self.mask_paths[idx],
            cv2.IMREAD_GRAYSCALE
        )

        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        mask = (mask > 127).astype(np.float32)

        if self.transform:

            augmented = self.transform(
                image=img,
                mask=mask
            )

            img = augmented["image"]

            mask = augmented["mask"].unsqueeze(0)

        else:

            img = torch.tensor(img).float().unsqueeze(0)/255.0

            mask = torch.tensor(mask).float().unsqueeze(0)

        return img, mask


# DATASET SPLITTING

full_dataset = KaggleNerveDataset(
    train_data_path,
    transform=None
)

indices = list(range(len(full_dataset)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=42
)

train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.1765,
    random_state=42
)

train_dataset = KaggleNerveDataset(
    train_data_path,
    transform=train_transform
)

val_dataset = KaggleNerveDataset(
    train_data_path,
    transform=val_transform
)

test_dataset = KaggleNerveDataset(
    train_data_path,
    transform=val_transform
)

train_dataset = torch.utils.data.Subset(
    train_dataset,
    train_idx
)

val_dataset = torch.utils.data.Subset(
    val_dataset,
    val_idx
)

test_dataset = torch.utils.data.Subset(
    test_dataset,
    test_idx
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# BASELINE MODELS

# 1. U-NET

class UNetBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, x):

        return torch.sigmoid(self.model(x))

# 2. RESUNET

class ResUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, x):

        return torch.sigmoid(self.model(x))

# 3. UNET++

class UNetPlusPlus(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, x):

        return torch.sigmoid(self.model(x))


# 4. TRANSUNET

class TransUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.Unet(
            encoder_name="tu-vit_base_patch16_224",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, x):

        return torch.sigmoid(self.model(x))

# 5. NNU-NET STYLE

class NNUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, x):

        return torch.sigmoid(self.model(x))

# PROPOSED MODEL

class EdgeFusionAttentionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.denoiser = DiffusionDenoiser()

        self.edge_extractor = EdgeExtractor()

        self.segmentor = smp.Unet(

            encoder_name="resnet34",

            encoder_weights="imagenet",

            in_channels=3,

            classes=1,

            decoder_attention_type="scse"

        )

    def forward(self, x):

        x_denoised = self.denoiser(x)

        x_edges = self.edge_extractor(x_denoised)

        fused = torch.cat([
            x,
            x_denoised,
            x_edges
        ], dim=1)

        mask = self.segmentor(fused)

        return torch.sigmoid(mask)


# LOSS FUNCTIONS

class DiceLoss(nn.Module):

    def __init__(self, smooth=1):

        super().__init__()

        self.smooth = smooth

    def forward(self, preds, targets):

        preds = preds.contiguous()
        targets = targets.contiguous()

        intersection = (preds * targets).sum(dim=(2,3))

        dice = (

            2. * intersection + self.smooth

        ) / (

            preds.sum(dim=(2,3))
            + targets.sum(dim=(2,3))
            + self.smooth

        )

        return 1 - dice.mean()

dice_loss = DiceLoss()

bce_loss = nn.BCELoss()

def combined_loss(preds, targets):

    bce = bce_loss(preds, targets)

    dice = dice_loss(preds, targets)

    return 0.5*bce + 0.5*dice

# METRICS

def calculate_metrics(preds, targets):

    preds = (preds > 0.5).float()

    preds_flat = preds.cpu().numpy().flatten()

    targets_flat = targets.cpu().numpy().flatten()

    dice = (

        2*(preds_flat * targets_flat).sum()

    ) / (

        preds_flat.sum()
        + targets_flat.sum()
        + 1e-8

    )

    iou = jaccard_score(
        targets_flat,
        preds_flat,
        average='binary'
    )

    precision = precision_score(
        targets_flat,
        preds_flat,
        zero_division=0
    )

    recall = recall_score(
        targets_flat,
        preds_flat,
        zero_division=0
    )

    f1 = f1_score(
        targets_flat,
        preds_flat,
        zero_division=0
    )

    accuracy = accuracy_score(
        targets_flat,
        preds_flat
    )

    return {

        "dice": dice,

        "iou": iou,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "accuracy": accuracy

    }

# TRAINING FUNCTION

def train_model(model,
                model_name,
                train_loader,
                val_loader,
                epochs=50):

    print(f"\nTraining {model_name}")

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5
    )

    best_dice = 0

    train_losses = []
    val_losses = []

    train_dices = []
    val_dices = []

    for epoch in range(epochs):

        # TRAINING

        model.train()

        train_loss = 0
        train_dice = 0

        for imgs, masks in train_loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            preds = model(imgs)

            loss = combined_loss(preds, masks)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            metrics = calculate_metrics(
                preds.detach(),
                masks.detach()
            )

            train_dice += metrics["dice"]

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # VALIDATION

        model.eval()

        val_loss = 0
        val_dice = 0

        with torch.no_grad():

            for imgs, masks in val_loader:

                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)

                loss = combined_loss(preds, masks)

                val_loss += loss.item()

                metrics = calculate_metrics(
                    preds,
                    masks
                )

                val_dice += metrics["dice"]

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(f"""
Epoch [{epoch+1}/{epochs}]

Train Loss: {train_loss:.4f}

Validation Loss: {val_loss:.4f}

Train Dice: {train_dice:.4f}

Validation Dice: {val_dice:.4f}
""")

        # SAVE BEST MODEL

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                f"{model_name}.pth"
            )

            print("Best model saved!")

    history = {

        "train_losses": train_losses,

        "val_losses": val_losses,

        "train_dices": train_dices,

        "val_dices": val_dices

    }

    return model, history


# TEST EVALUATION FUNCTION

def evaluate_model(model,
                   model_name,
                   test_loader):

    print(f"\nEvaluating {model_name}")

    model.eval()

    all_metrics = []

    all_preds = []
    all_targets = []

    inference_times = []

    with torch.no_grad():

        for imgs, masks in test_loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            start = time.time()

            preds = model(imgs)

            end = time.time()

            inference_times.append(end-start)

            metrics = calculate_metrics(
                preds,
                masks
            )

            all_metrics.append(metrics)

            preds_bin = (preds > 0.5).float()

            all_preds.extend(
                preds_bin.cpu().numpy().flatten()
            )

            all_targets.extend(
                masks.cpu().numpy().flatten()
            )

    avg_metrics = {}

    for key in all_metrics[0].keys():

        avg_metrics[key] = np.mean(
            [m[key] for m in all_metrics]
        )

    print("\n===== RESULTS =====")

    for k,v in avg_metrics.items():

        print(f"{k}: {v:.4f}")

    print(
        f"Average Inference Time:"
        f"{np.mean(inference_times):.4f}"
    )

    # CONFUSION MATRIX

    cm = confusion_matrix(
        all_targets,
        all_preds
    )

    plt.figure(figsize=(6,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.show()

    # ROC CURVE

    fpr, tpr, _ = roc_curve(
        all_targets,
        all_preds
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.4f}"
    )

    plt.plot([0,1],[0,1],'--')

    plt.legend()

    plt.title(f"{model_name} ROC Curve")

    plt.show()

    return avg_metrics

# VISUALIZATION FUNCTION

def visualize_predictions(model, loader):

    model.eval()

    imgs, masks = next(iter(loader))

    imgs = imgs.to(device)

    with torch.no_grad():

        preds = model(imgs)

    preds = (preds > 0.5).float()

    fig, axes = plt.subplots(3,3, figsize=(12,12))

    for i in range(3):

        axes[i,0].imshow(
            imgs[i].cpu().squeeze(),
            cmap='gray'
        )

        axes[i,0].set_title("Input")

        axes[i,1].imshow(
            preds[i].cpu().squeeze(),
            cmap='jet'
        )

        axes[i,1].set_title("Prediction")

        axes[i,2].imshow(
            masks[i].squeeze(),
            cmap='gray'
        )

        axes[i,2].set_title("Ground Truth")

    plt.tight_layout()

    plt.show()

# INITIALIZE ALL MODELS

models = {

    "UNet": UNetBaseline(),

    "ResUNet": ResUNet(),

    "UNetPlusPlus": UNetPlusPlus(),

    "TransUNet": TransUNet(),

    "NNUNet": NNUNet(),

    "EdgeFusionAttentionUNet":
        EdgeFusionAttentionUNet()

}

# TRAIN ALL MODELS

all_results = {}

all_histories = {}

for model_name, model in models.items():

    trained_model, history = train_model(

        model=model,

        model_name=model_name,

        train_loader=train_loader,

        val_loader=val_loader,

        epochs=50

    )

    results = evaluate_model(

        trained_model,

        model_name,

        test_loader

    )

    visualize_predictions(
        trained_model,
        test_loader
    )

    all_results[model_name] = results

    all_histories[model_name] = history

# FINAL COMPARISON TABLE

comparison_table = []

for model_name, metrics in all_results.items():

    comparison_table.append({

        "Model": model_name,

        "Dice":
            round(metrics["dice"]*100,2),

        "IoU":
            round(metrics["iou"]*100,2),

        "Precision":
            round(metrics["precision"]*100,2),

        "Recall":
            round(metrics["recall"]*100,2),

        "F1":
            round(metrics["f1"]*100,2),

        "Accuracy":
            round(metrics["accuracy"]*100,2)

    })

results_df = pd.DataFrame(comparison_table)

print(results_df)

# SAVE RESULTS


results_df.to_csv(
    "all_model_results.csv",
    index=False
)

print("Results Saved!")

# BAR CHART COMPARISON

plt.figure(figsize=(12,6))

plt.bar(
    results_df["Model"],
    results_df["Dice"]
)

plt.xticks(rotation=20)

plt.ylabel("Dice Score")

plt.title("Dice Score Comparison")

plt.show()

# TRAINING CURVES

for model_name, history in all_histories.items():

    plt.figure(figsize=(10,5))

    plt.plot(
        history["train_losses"],
        label='Train Loss'
    )

    plt.plot(
        history["val_losses"],
        label='Validation Loss'
    )

    plt.title(model_name)

    plt.legend()

    plt.show()

# STATISTICAL ANALYSIS

dice_scores = np.array([

    metrics["dice"]

    for metrics in all_results.values()

])

mean_dice = np.mean(dice_scores)

std_dice = np.std(dice_scores)

print("\n===== STATISTICAL ANALYSIS =====")

print(f"Mean Dice Score: {mean_dice:.4f}")

print(f"Std Dice Score: {std_dice:.4f}")

confidence = 0.95

n = len(dice_scores)

h = sem(dice_scores) * t.ppf(
    (1 + confidence)/2,
    n - 1
)

print(
    f"95% Confidence Interval:"
    f"[{mean_dice-h:.4f}, {mean_dice+h:.4f}]"
)

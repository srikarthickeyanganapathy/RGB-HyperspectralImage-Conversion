import torch
import cv2
import numpy as np
import scipy.io as sio
import os
import glob
from models import ResNetGenerator

# --- CONFIG ---
INPUT_IMAGE_FOLDER = "input_images" # Create this and put JPGs here
OUTPUT_MAT_FOLDER = "output_mats"
MODEL_PATH = "checkpoints/netG_final.pth"
BANDS = 224 # Match your training data (224 spectral bands)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_MAT_FOLDER, exist_ok=True)

def run_inference():
    model = ResNetGenerator(output_nc=BANDS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    images = glob.glob(os.path.join(INPUT_IMAGE_FOLDER, "*.jpg"))
    print(f"Found {len(images)} images to convert.")

    for img_path in images:
        # Load & Preprocess
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = (img.astype(np.float32) / 255.0) * 2.0 - 1.0 # Norm to -1 to 1
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            fake_hs = model(tensor)

        # Post-process
        hs_data = fake_hs.squeeze().permute(1, 2, 0).cpu().numpy()
        hs_data = (hs_data + 1.0) / 2.0 # Denorm back to 0-1 range

        # Save
        name = os.path.basename(img_path).replace(".jpg", ".mat")
        save_path = os.path.join(OUTPUT_MAT_FOLDER, name)
        sio.savemat(save_path, {'cube': hs_data})
        print(f"Generated: {save_path}")

if __name__ == "__main__":
    run_inference()
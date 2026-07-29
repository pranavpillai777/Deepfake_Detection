import os
import torch

# Corrected path using 'checkpoints' (plural)
checkpoint_path = r"C:\Users\Toshiba\Desktop\LY_PROJECT\checkpoints\mobilenet_v2_deepfake.pth"

if os.path.exists(checkpoint_path):
    print(f"✅ Found checkpoint at: {os.path.abspath(checkpoint_path)}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle wrapped dictionaries if applicable
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
    print("\n--- First Layer Weight Shape ---")
    if 'features.0.0.weight' in state_dict:
        print(f"features.0.0.weight shape: {state_dict['features.0.0.weight'].shape}")
    else:
        print("⚠️ 'features.0.0.weight' not found. Available keys sample:")
        print(list(state_dict.keys())[:5])

    print("\n--- Final Classifier Weight Shape ---")
    if 'classifier.1.weight' in state_dict:
        print(f"classifier.1.weight shape: {state_dict['classifier.1.weight'].shape}")
    else:
        print("⚠️ 'classifier.1.weight' not found.")
else:
    print(f"❌ Checkpoint still not found at: {checkpoint_path}")
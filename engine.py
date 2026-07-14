import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def process_image(image_path, output_path, model_path="sam_vit_b_01ec64.pth"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}. Check the path.")
    
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    print("Loading SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    sam.to(device=device)

    
    print(f"Running inference on {device} (this may take a few seconds)...")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_rgb)

    
    print(f"Found {len(masks)} segments. Drawing overlays...")
    overlay = image.copy()
    
    
    for mask_data in masks:
        segment_mask = mask_data['segmentation']
        
        
        color = np.random.randint(0, 255, 3).tolist() 
        

        overlay[segment_mask] = color 


    alpha = 0.5 
    final_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


    cv2.imwrite(output_path, final_image)
    print(f"Success! Masked image saved to {output_path}")

if __name__ == "__main__":
    process_image("test.png", "output.png")
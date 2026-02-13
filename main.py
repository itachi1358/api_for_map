import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from skimage import morphology

app = FastAPI(title="Satellite Image Segmentation API")

# --- YOUR LOGIC CONSTANTS ---
THRESH_VEG = 0.15 
THRESH_WATER = 0.15

def get_land_cover_colors():
    return {
        0: {'name': 'Urban', 'color': [220, 220, 220]},
        1: {'name': 'Vegetation', 'color': [34, 139, 34]},
        2: {'name': 'Water', 'color': [65, 105, 225]},
        3: {'name': 'Barren', 'color': [139, 69, 19]}
    }

def robust_segmentation(img_array):
    img = img_array.astype(float) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    vari = (g - r) / (g + r - b + 1e-6)
    ndwi = (b - r) / (b + r + 1e-6)
    
    max_val = np.max(img, axis=2)
    min_val = np.min(img, axis=2)
    saturation = (max_val - min_val) / (max_val + 1e-6)
    intensity = np.mean(img, axis=2)
    
    h, w = r.shape
    seg_map = np.zeros((h, w), dtype=int)
    
    mask_veg = (vari > THRESH_VEG) & (g > r) & (g > b * 0.8)
    mask_water = (ndwi > THRESH_WATER) & (b > r) & (intensity < 0.6) & (~mask_veg)
    mask_remaining = (~mask_veg) & (~mask_water)
    mask_urban = mask_remaining & (saturation < 0.15)
    mask_soil = mask_remaining & (saturation >= 0.15)
    
    seg_map[mask_urban] = 0
    seg_map[mask_veg] = 1
    seg_map[mask_water] = 2
    seg_map[mask_soil] = 3
    return seg_map

def apply_colormap(seg_map):
    colors = get_land_cover_colors()
    h, w = seg_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, props in colors.items():
        colored[seg_map == cid] = props['color']
    return colored

@app.get("/")
async def f():
    return "Hello from Results"


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # 1. Read the uploaded image into memory
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    img_array = np.array(image)
    
    # 2. Process
    raw_map = robust_segmentation(img_array)
    # Morphological cleaning
    clean_map = morphology.opening(raw_map, morphology.disk(3))
    clean_map = morphology.closing(clean_map, morphology.disk(3))
    
    # 3. Colorize
    colored_result = apply_colormap(clean_map)
    output_img = Image.fromarray(colored_result)
    
    # 4. Save result to a byte buffer (RAM) to send back
    img_byte_arr = io.BytesIO()
    output_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
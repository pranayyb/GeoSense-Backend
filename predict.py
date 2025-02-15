import os
import requests
import cv2
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from segment_anything import sam_model_registry, SamPredictor
from pydantic import BaseModel
import supervision as sv
from io import BytesIO
from PIL import Image
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

MODEL_TYPE = "vit_h"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

HOME = os.getcwd()
WEIGHTS_DIR = os.path.join(HOME, "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

os.makedirs(WEIGHTS_DIR, exist_ok=True)


def download_checkpoint(url, save_path):

    if os.path.isfile(save_path):
        print("Checkpoint already exists.")
        return

    print("Downloading model checkpoint... This may take a few minutes.")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        raise RuntimeError(
            f"Failed to download checkpoint. HTTP Status: {response.status_code}"
        )


download_checkpoint(CHECKPOINT_URL, CHECKPOINT_PATH)

try:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=device)
    mask_predictor = SamPredictor(sam)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


@app.post("/segment/")
async def segment_image(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        box = np.array([x1, y1, x2, y2])

        mask_predictor.set_image(image_np)
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=False)

        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]
        mask = detections.mask[0]

        mask_area = np.sum(mask)
        mask_percentage = (mask_area / (mask.shape[0] * mask.shape[1])) * 100

        return {
            "Mask Area (pixels)": int(mask_area),
            "Mask Area Percentage": round(mask_percentage, 4),
            "Bounding Box": {
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
                "Width": width,
                "Height": height,
            },
        }
    except Exception as e:
        logger.error(f"Error processing segmentation: {e}")
        return JSONResponse(
            status_code=500, content={"message": "Segmentation failed", "error": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)

import os
import requests
import cv2
import torch
import numpy as np
import uvicorn
import supervision as sv
import logging
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from segment_anything import sam_model_registry, SamPredictor
from pydantic import BaseModel
from io import BytesIO
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

MODEL_TYPE = "vit_h"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

HOME = os.getcwd()
WEIGHTS_DIR = os.path.join(HOME, "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

os.makedirs(WEIGHTS_DIR, exist_ok=True)


def download_checkpoint(url, save_path):
    try:
        if os.path.isfile(save_path):
            logger.info("Checkpoint already exists.")
            return

        logger.info("Downloading model checkpoint... This may take a few minutes.")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Download complete!")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise RuntimeError(f"Failed to download checkpoint: {e}")


try:
    download_checkpoint(CHECKPOINT_URL, CHECKPOINT_PATH)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=device)
    mask_predictor = SamPredictor(sam)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
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
    request: Request,
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
):
    try:
        logger.info(f"Request received from {request.client.host} for segmentation.")

        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        logger.info("Image successfully loaded and converted.")

        box = np.array([x1, y1, x2, y2])
        mask_predictor.set_image(image_np)
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=False)

        logger.info("Segmentation prediction completed.")

        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]
        mask = detections.mask[0]

        mask_area = np.sum(mask)
        mask_percentage = (mask_area / (mask.shape[0] * mask.shape[1])) * 100

        mask = (mask * 255).astype(np.uint8)
        green_overlay = np.zeros_like(image_np, dtype=np.uint8)
        green_overlay[:, :, 1] = mask  # Apply mask to the green channel
        masked_image = cv2.addWeighted(image_np, 1.0, green_overlay, 0.5, 0)
        masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode(
            ".png", masked_image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )

        logger.info(
            f"Segmentation completed successfully. Sending response to {request.client.host}."
        )

        return Response(
            content=encoded_image.tobytes(),
            media_type="image/png",
            headers={
                "X-Mask-Area": str(int(mask_area)),
                "X-Mask-Percentage": str(round(mask_percentage, 4)),
            },
        )
    except Exception as e:
        logger.error(f"Error processing segmentation: {e}")
        return JSONResponse(
            status_code=500, content={"message": "Segmentation failed", "error": str(e)}
        )


if __name__ == "__main__":
    try:
        logger.info("Starting FastAPI server...")
        uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Error starting FastAPI server: {e}")

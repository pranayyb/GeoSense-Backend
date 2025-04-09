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
from fastapi.responses import JSONResponse
from segment_anything import sam_model_registry, SamPredictor
from pydantic import BaseModel
from io import BytesIO
import io
from PIL import Image

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


def mask_to_svg_string(mask):
    """Converts a binary mask to an SVG string representation."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape

    svg_header = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    paths = []

    for contour in contours:
        if len(contour) > 0:
            path_data = (
                "M "
                + " L ".join(f"{point[0][0]},{point[0][1]}" for point in contour)
                + " Z"
            )
            paths.append(
                f'<path d="{path_data}" fill="green" stroke="black" stroke-width="1"/>'
            )

    svg_footer = "</svg>"
    return svg_header + "".join(paths) + svg_footer


def svg_to_blob(svg_content: str) -> io.BytesIO:
    blob = io.BytesIO(svg_content.encode("utf-8"))  # Convert SVG string to bytes
    return blob


@app.post("/img/")
async def segment_image(
    request: Request,
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    scaleVal: float = Form(...),
):
    try:
        logger.info(f"Request received from {request.client.host} for segmentation.")

        # Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Run segmentation
        box = np.array([x1, y1, x2, y2])
        mask_predictor.set_image(image_np)
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=False)

        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]

        if len(detections.mask) == 0:
            return JSONResponse(
                status_code=400, content={"message": "No valid mask found."}
            )

        mask = detections.mask[0]
        mask_area = int(np.sum(mask))
        mask_percentage = round((mask_area / (mask.shape[0] * mask.shape[1])) * 100, 4)

        mask = (mask * 255).astype(np.uint8)
        green_overlay = np.zeros_like(image_np, dtype=np.uint8)
        green_overlay[:, :, 1] = mask
        masked_image = cv2.addWeighted(image_np, 1.0, green_overlay, 0.5, 0)
        masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

        # Encode to PNG
        _, encoded_image = cv2.imencode(
            ".png", masked_image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
        image_blob = io.BytesIO(encoded_image.tobytes())

        logger.info(f"Segmentation completed. Sending JSON with image blob.")
        # print(scaleVal)
        final_area_accurate = (scaleVal * scaleVal * mask_area) / 1.5136
        final_area_accurate_acres = final_area_accurate / 4046.856422
        return JSONResponse(
            content={
                "area": final_area_accurate_acres,
                "mask_area": mask_area,
                "mask_percentage": mask_percentage,
                "image_blob": image_blob.getvalue().hex(),
            }
        )
    except Exception as e:
        logger.error(f"Error processing segmentation: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Segmentation failed", "error": str(e)},
        )


@app.post("/svg/")
async def segment_image(
    request: Request,
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    scaleVal: float = Form(...),
):
    try:
        logger.info(f"Request received from {request.client.host} for segmentation.")

        # Read and process image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        logger.info("Image successfully loaded and converted.")

        # Define bounding box
        box = np.array([x1, y1, x2, y2])
        mask_predictor.set_image(image_np)
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=False)

        logger.info("Segmentation prediction completed.")

        # Process mask
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]

        if len(detections.mask) == 0:
            return JSONResponse(
                status_code=400, content={"message": "No valid mask found."}
            )

        mask = detections.mask[0]
        mask_binary = (mask * 255).astype(np.uint8)

        # Calculate mask area
        mask_area = int(np.sum(mask))  # Total number of pixels in the mask
        mask_percentage = round((mask_area / (mask.shape[0] * mask.shape[1])) * 100, 4)

        # Convert mask to SVG string
        mask_svg = mask_to_svg_string(mask_binary)
        # print(mask_svg)
        svg_blob = svg_to_blob(mask_svg)
        # print(svg_blob.getvalue())

        logger.info(
            f"Segmentation completed successfully. Sending JSON response to {request.client.host}."
        )
        final_area_accurate = (scaleVal * scaleVal * mask_area) / 1.5136
        final_area_accurate_acres = final_area_accurate / 4046.856422
        return JSONResponse(
            content={
                "area": final_area_accurate_acres,
                "svg": f"{svg_blob.getvalue()}",
                "mask_area": mask_area,
                "mask_percentage": mask_percentage,
            }
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

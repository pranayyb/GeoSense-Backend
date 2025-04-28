from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import numpy as np
import io
from PIL import Image
import supervision as sv
from utils import mask_to_svg_string, svg_to_blob
from config import SCALE_NORMALIZER, ACRES_CONVERSION
from io import BytesIO
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
@router.post("/svg/")
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
        from main import mask_predictor

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

        if len(detections.mask) == 0:
            return JSONResponse(
                status_code=400, content={"message": "No valid mask found."}
            )

        mask = detections.mask[0]
        mask_binary = (mask * 255).astype(np.uint8)
        mask_area = int(np.sum(mask))
        mask_percentage = round((mask_area / (mask.shape[0] * mask.shape[1])) * 100, 4)

        mask_svg = mask_to_svg_string(mask_binary)
        # print(mask_svg)
        svg_blob = svg_to_blob(mask_svg)
        # print(svg_blob.getvalue())

        logger.info(
            f"Segmentation completed successfully. Sending JSON response to {request.client.host}."
        )
        final_area_accurate = (scaleVal * scaleVal * mask_area) / SCALE_NORMALIZER
        final_area_accurate_acres = final_area_accurate / ACRES_CONVERSION
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

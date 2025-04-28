from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image
import supervision as sv
from config import SCALE_NORMALIZER, ACRES_CONVERSION
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/")
@router.post("/img/")
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
        logger.info(
            f"Parameters: x1={x1}, y1={y1}, x2={x2}, y2={y2}, width={width}, height={height}, scaleVal={scaleVal}"
        )

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

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

        _, encoded_image = cv2.imencode(
            ".png", masked_image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
        image_blob = io.BytesIO(encoded_image.tobytes())

        logger.info(f"Segmentation completed. Sending JSON with image blob.")
        # print(scaleVal)
        final_area_accurate = (scaleVal * scaleVal * mask_area) / SCALE_NORMALIZER
        final_area_accurate_acres = final_area_accurate / ACRES_CONVERSION
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
        logger.error(f"Request details: {request.headers}")
        return JSONResponse(
            status_code=500,
            content={"message": "Segmentation failed", "error": str(e)},
        )

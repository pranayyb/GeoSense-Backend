from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import json
import numpy as np
import io
import cv2
from io import BytesIO
from PIL import Image
import supervision as sv
from utils import mask_to_svg_string, svg_to_blob
from config import SCALE_NORMALIZER, ACRES_CONVERSION
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/")
@router.post("/bulk/")
async def bulk_process_images(
    request: Request,
    files: list[UploadFile] = File(...),
    params: str = Form(...),
):
    try:
        from main import mask_predictor

        logger.info(f"Bulk processing request received from {request.client.host}")
        logger.info(f"Processing {len(files)} images")

        try:
            image_params = json.loads(params)
            if not isinstance(image_params, list) or len(image_params) != len(files):
                return JSONResponse(
                    status_code=400,
                    content={
                        "message": f"Number of parameter sets ({len(image_params)}) doesn't match number of files ({len(files)})"
                    },
                )
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"message": "Invalid JSON format for parameters"},
            )

        logger.info(f"Received parameters for {len(image_params)} images")

        results = []

        for i, (file, img_param) in enumerate(zip(files, image_params)):
            try:
                x1 = img_param.get("x1")
                y1 = img_param.get("y1")
                x2 = img_param.get("x2")
                y2 = img_param.get("y2")
                width = img_param.get("width")
                height = img_param.get("height")
                scaleVal = img_param.get("scaleVal")
                format_type = img_param.get("format", "image")
                required_params = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "scaleVal": scaleVal,
                }
                missing_params = [k for k, v in required_params.items() if v is None]

                if missing_params:
                    results.append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "message": f"Missing required parameters: {', '.join(missing_params)}",
                        }
                    )
                    continue

                logger.info(f"Processing image {i+1}/{len(files)}: {file.filename}")
                logger.info(
                    f"Parameters: x1={x1}, y1={y1}, x2={x2}, y2={y2}, width={width}, height={height}, scaleVal={scaleVal}"
                )

                image_bytes = await file.read()
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image_np = np.array(image)

                box = np.array([x1, y1, x2, y2])
                mask_predictor.set_image(image_np)
                masks, scores, logits = mask_predictor.predict(
                    box=box, multimask_output=False
                )

                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks), mask=masks
                )
                detections = detections[detections.area == np.max(detections.area)]

                if len(detections.mask) == 0:
                    results.append(
                        {
                            "filename": file.filename,
                            "success": False,
                            "message": "No valid mask found",
                        }
                    )
                    continue

                mask = detections.mask[0]
                mask_area = int(np.sum(mask))
                mask_percentage = round(
                    (mask_area / (mask.shape[0] * mask.shape[1])) * 100, 4
                )
                final_area_accurate = (
                    scaleVal * scaleVal * mask_area
                ) / SCALE_NORMALIZER
                final_area_accurate_acres = final_area_accurate / ACRES_CONVERSION

                result = {
                    "filename": file.filename,
                    "success": True,
                    "area": final_area_accurate_acres,
                    "mask_area": mask_area,
                    "mask_percentage": mask_percentage,
                }

                if format_type.lower() == "svg":
                    mask_binary = (mask * 255).astype(np.uint8)
                    mask_svg = mask_to_svg_string(mask_binary)
                    svg_blob = svg_to_blob(mask_svg)
                    result["svg"] = svg_blob.getvalue().decode("utf-8")
                else:  # Default to image
                    mask = (mask * 255).astype(np.uint8)
                    green_overlay = np.zeros_like(image_np, dtype=np.uint8)
                    green_overlay[:, :, 1] = mask
                    masked_image = cv2.addWeighted(image_np, 1.0, green_overlay, 0.5, 0)
                    masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

                    _, encoded_image = cv2.imencode(
                        ".png", masked_image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]
                    )
                    image_blob = io.BytesIO(encoded_image.tobytes())
                    result["image_blob"] = image_blob.getvalue().hex()

                results.append(result)
                logger.info(f"Successfully processed image {i+1}: {file.filename}")

            except Exception as e:
                logger.error(f"Error processing image {file.filename}: {e}")
                results.append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "message": f"Processing failed: {str(e)}",
                    }
                )
            finally:
                mask_predictor.reset_image()

        logger.info(f"Bulk processing completed. Processed {len(results)} images.")
        return JSONResponse(
            content={
                "total_processed": len(results),
                "successful": sum(1 for r in results if r.get("success", False)),
                "results": results,
            }
        )
    except Exception as e:
        logger.error(f"Error in bulk processing: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Bulk processing failed", "error": str(e)},
        )

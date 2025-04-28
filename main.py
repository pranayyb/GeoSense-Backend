import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import DEVICE, MODEL_TYPE, CHECKPOINT_URL, CHECKPOINT_PATH, WEIGHTS_DIR
from utils import download_checkpoint
from segment_anything import sam_model_registry, SamPredictor
from api import img, svg, bulk

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(WEIGHTS_DIR, exist_ok=True)
download_checkpoint(CHECKPOINT_URL, CHECKPOINT_PATH)
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

app.include_router(img.router, prefix="/img", tags=["img"])
app.include_router(svg.router, prefix="/svg", tags=["svg"])
app.include_router(bulk.router, prefix="/bulk", tags=["bulk"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)

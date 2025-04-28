import os
import requests
import logging
import io
import cv2

logger = logging.getLogger(__name__)


def download_checkpoint(url: str, save_path: str):
    try:
        if os.path.isfile(save_path):
            logger.info("Checkpoint already exists.")
            return

        logger.info("Downloading model checkpoint...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise RuntimeError(f"Failed to download checkpoint: {e}")


def mask_to_svg_string(mask):
    """Converts a binary mask to SVG string."""
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
    """Converts SVG content to binary blob."""
    return io.BytesIO(svg_content.encode("utf-8"))

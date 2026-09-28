# ocr_engine.py
#
# Stage 1: local OCR draft, adapted from PrismAPI's app/services/ocr_service.py
# (same RapidOCR engine + CLAHE/upscale preprocessing), trimmed to a single
# in-memory image with no PDF/page/DB handling.

import logging

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

logger = logging.getLogger(__name__)

_ocr_engine = None


def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        if RapidOCR is None:
            raise RuntimeError("rapidocr-onnxruntime is not installed")
        _ocr_engine = RapidOCR()
        logger.info("RapidOCR engine initialized")
    return _ocr_engine


def _preprocess(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement + 2x upscale, same as PrismAPI's OCR preprocessing."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        h, w = enhanced.shape
        resized = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.warning("OCR preprocessing failed, using original image: %s", e)
        return img


def extract_draft_text(img: np.ndarray) -> str:
    """Run local OCR on a BGR image and return the detected text, joined in reading order."""
    engine = _get_engine()
    processed = _preprocess(img)

    result, _elapse = engine(processed)
    if not result:
        return ""

    lines = [text.strip() for _bbox, text, _conf in result if text and text.strip()]
    return " ".join(lines)

import sys
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pathlib import Path

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Extract text from an image.
        Returns a list of dicts:
        [
            {
                "text": str,
                "confidence": float (0.0 to 1.0),
                "bbox": list of [x, y] coordinates
            },
            ...
        ]
        """
        pass

class EasyOCREngine(BaseOCREngine):
    """EasyOCR implementation using PyTorch deep learning models."""
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        if sys.platform.startswith("win"):
            # Prevent Windows cp1252 stdout encoding crash during progress bar prints
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass
                
        import easyocr
        if languages is None:
            languages = ['en']
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        if isinstance(image, (str, Path)):
            image = str(image)
        
        raw_results = self.reader.readtext(image)
        formatted_results = []
        
        for bbox, text, confidence in raw_results:
            # Convert bbox numpy types to standard python lists/floats
            bbox_list = [[int(pt[0]), int(pt[1])] for pt in bbox]
            formatted_results.append({
                "text": text.strip(),
                "confidence": round(float(confidence), 4),
                "bbox": bbox_list
            })
            
        return formatted_results

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR implementation via pytesseract."""
    
    def __init__(self, tesseract_cmd: str = None, psm: int = 6):
        import pytesseract
        self.pytesseract = pytesseract
        self.psm = psm
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif sys.platform.startswith("win"):
            default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_win_path):
                pytesseract.pytesseract.tesseract_cmd = default_win_path

    def extract_text(self, image: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        import cv2
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image
            
        config = f"--psm {self.psm}"
        data = self.pytesseract.image_to_data(img, config=config, output_type=self.pytesseract.Output.DICT)
        
        formatted_results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            if text and conf > 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                formatted_results.append({
                    "text": text,
                    "confidence": round(conf / 100.0, 4),
                    "bbox": bbox
                })
        return formatted_results

def get_ocr_engine(engine_name: str = "easyocr", **kwargs) -> BaseOCREngine:
    """Factory function to instantiate the requested OCR engine."""
    engine_name = engine_name.lower()
    if engine_name == "easyocr":
        return EasyOCREngine(**kwargs)
    elif engine_name in ("tesseract", "pytesseract"):
        return TesseractEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported OCR engine '{engine_name}'. Choose 'easyocr' or 'tesseract'.")

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union

from preprocess import load_image, preprocess
from ocr_engine import get_ocr_engine

def annotate_image(image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and recognized text overlay on the image."""
    annotated = image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        
    for res in results:
        bbox = res["bbox"]
        text = res["text"]
        conf = res["confidence"]
        
        # Extract rectangle bounding points
        pts = np.array(bbox, dtype=np.int32)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        
        # Draw bounding rectangle
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw background label box
        label = f"{text} ({int(conf * 100)}%)"
        (font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x_min, max(0, y_min - font_h - 6)),
            (x_min + font_w, y_min),
            (0, 255, 0),
            -1
        )
        # Put white text
        cv2.putText(
            annotated,
            label,
            (x_min, max(12, y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    return annotated

def run_ocr_pipeline(
    image_path: Union[str, Path],
    engine_name: str = "easyocr",
    apply_preprocessing: bool = True,
    deskew: bool = False,
    output_preprocessed_path: str = None,
    output_annotated_path: str = None
) -> Tuple[str, List[Dict[str, Any]], np.ndarray]:
    """
    Executes end-to-end OCR pipeline:
    1. Loads raw image
    2. Applies OpenCV preprocessing (optional)
    3. Runs OCR engine (EasyOCR or Tesseract)
    4. Annotates output visualization image
    5. Returns extracted plain text, detailed structured results, and preprocessed image.
    """
    raw_image = load_image(image_path)
    
    if apply_preprocessing:
        processed_image = preprocess(raw_image, deskew_flag=deskew)
    else:
        processed_image = raw_image

    engine = get_ocr_engine(engine_name)
    results = engine.extract_text(processed_image)

    # Save outputs if paths provided
    if output_preprocessed_path:
        cv2.imwrite(output_preprocessed_path, processed_image)
        
    if output_annotated_path:
        annotated_img = annotate_image(raw_image, results)
        cv2.imwrite(output_annotated_path, annotated_img)

    extracted_lines = [res["text"] for res in results]
    full_text = "\n".join(extracted_lines)

    return full_text, results, processed_image

def main():
    parser = argparse.ArgumentParser(description="Python-based OpenCV + Deep Learning OCR Pipeline")
    parser.add_argument("image", help="Path to input image file")
    parser.add_argument("-o", "--output", help="Path to save preprocessed output image", default="samples/preprocessed.png")
    parser.add_argument("-a", "--annotated", help="Path to save annotated bounding-box image", default="samples/annotated.png")
    parser.add_argument("-e", "--engine", choices=["easyocr", "tesseract"], default="easyocr", help="OCR engine backend (default: easyocr)")
    parser.add_argument("--json", action="store_true", help="Output full JSON formatted extraction metadata")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OpenCV preprocessing stage")
    parser.add_argument("--deskew", action="store_true", help="Enable orientation deskewing")

    args = parser.parse_args()

    full_text, results, _ = run_ocr_pipeline(
        image_path=args.image,
        engine_name=args.engine,
        apply_preprocessing=not args.no_preprocess,
        deskew=args.deskew,
        output_preprocessed_path=args.output,
        output_annotated_path=args.annotated
    )

    print("\n================ Extracted OCR Text ================")
    print(full_text if full_text.strip() else "[No text detected]")
    print("====================================================\n")

    if args.json:
        print("Structured Results (JSON):")
        print(json.dumps(results, indent=2))

    print(f"\n[+] Saved preprocessed image to: {args.output}")
    print(f"[+] Saved annotated visualization to: {args.annotated}")

if __name__ == "__main__":
    main()

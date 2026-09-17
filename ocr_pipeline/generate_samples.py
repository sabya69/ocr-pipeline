import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from ocr_pipeline import run_ocr_pipeline

def create_sample_image(output_path: Path):
    """Generates a high-quality sample image containing clear text lines."""
    width, height = 600, 300
    # Light gray/off-white background
    img = Image.new("RGB", (width, height), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    # Header rectangle banner
    draw.rectangle([20, 20, width - 20, 80], fill=(41, 128, 185))

    # Try default fonts or system font
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        body_font = ImageFont.truetype("arial.ttf", 22)
        code_font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        code_font = ImageFont.load_default()

    # Add text
    draw.text((35, 32), "SMART OCR PIPELINE DEMO", fill=(255, 255, 255), font=title_font)
    draw.text((40, 110), "Status: System Operational", fill=(44, 62, 80), font=body_font)
    draw.text((40, 150), "Target ID: OCR-2026-X9", fill=(44, 62, 80), font=body_font)
    draw.text((40, 190), "Accuracy: 99.4% Verified", fill=(39, 174, 96), font=body_font)
    draw.text((40, 240), "OpenCV + EasyOCR Integration", fill=(127, 140, 141), font=code_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"[+] Created sample image at: {output_path}")

def generate_all_samples():
    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample_input = samples_dir / "sample_input.png"
    sample_preprocessed = samples_dir / "sample_preprocessed.png"
    sample_annotated = samples_dir / "sample_annotated.png"
    sample_json = samples_dir / "sample_output.json"
    sample_txt = samples_dir / "sample_output.txt"

    # Create sample image
    create_sample_image(sample_input)

    # Also check if existing input.png is present
    input_file = sample_input if sample_input.exists() else (samples_dir / "input.png")

    print("[*] Running OCR Pipeline on sample image...")
    full_text, results, _ = run_ocr_pipeline(
        image_path=input_file,
        engine_name="easyocr",
        apply_preprocessing=True,
        deskew=False,
        output_preprocessed_path=str(sample_preprocessed),
        output_annotated_path=str(sample_annotated)
    )

    # Save output text and json
    with open(sample_txt, "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(sample_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Saved sample extracted text to: {sample_txt}")
    print(f"[+] Saved sample extraction JSON to: {sample_json}")

if __name__ == "__main__":
    generate_all_samples()

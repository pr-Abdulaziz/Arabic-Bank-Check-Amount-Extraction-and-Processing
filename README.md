# Arabic Bank Check Amount Extraction and Processing

This project focuses on extracting the **Arabic legal amount written on bank checks** and converting it into a clean, machine-usable representation.  
Based on the course document (`Term Project-ICS472.pdf`) and the project report (`Arabic_Check_Processing_Report.pdf`), the core idea is to combine **computer vision + OCR + Arabic text post-processing** to build a reliable end-to-end pipeline.

## Project Idea

The target problem is challenging because Arabic bank-check text can include:
1. handwritten and printed variations,
2. noisy or low-quality scans,
3. segmented tokens and shape variations in Arabic script.

The proposed solution is a multi-stage pipeline:
1. detect or localize the legal-amount text region on the check,
2. preprocess the cropped text (grayscale, thresholding, denoising, contrast),
3. run OCR (Tesseract/EasyOCR and optional custom CNN-assisted components),
4. normalize recognized Arabic tokens,
5. convert Arabic amount words into numeric value,
6. evaluate accuracy and robustness.

The report indicates strong performance on clean samples and lower performance on noisy images, which aligns with expected OCR behavior on real bank checks.

## Current Repository Contents

1. `CheckImages\`  
   Dataset of check images (`.tif`) used for experiments.
2. `ExampleAnnotations-BoundingBoxes.txt`  
   Sample bounding-box annotation in YOLO style (`class x_center y_center width height` normalized).
3. `ExampleAnnotations-LegalAmounts.txt`  
   Sample legal-amount annotation as tokenized Arabic words mapped to image names.
4. `Term Project-ICS472.pdf`  
   Course project brief and expected scope.
5. `Arabic_Check_Processing_Report.pdf`  
   Final report with methodology, implementation summary, and results.

## Project Notebooks

1. `notebooks\01_legal_and_courtesy_amount_extraction.ipynb`  
   Setup for legal/courtesy field extraction using YOLOv8.
2. `notebooks\02_courtesy_amount_recognition.ipynb`  
   Setup for courtesy amount recognition using CNN-BiLSTM-CTC.
3. `notebooks\03_legal_amount_recognition.ipynb`  
   Setup for legal amount recognition, preprocessing, and tokenization config.
4. `notebooks\04_final_verification.ipynb`  
   Setup for final legal/courtesy matching and evaluation metrics.

## Quick Start

1. Open notebooks in numerical order (`01` to `04`) and run setup cells in each notebook.
2. Confirm phase paths are detected and output folders are created under `data\`, `models\`, and `outputs\`.

## Full Implementation Plan

1. **Dataset organization**
   - Split train/validation/test sets.
   - Standardize annotation format for both region localization and legal amount text.
2. **Region extraction**
   - Train or fine-tune a detector for legal-amount field localization.
   - Export cropped legal-amount regions for OCR.
3. **OCR and text normalization**
   - Use EasyOCR/Tesseract baseline.
   - Normalize spacing, ligatures, token boundaries, and common recognition errors.
4. **Arabic amount parsing**
   - Build rule-based parser for Arabic number words (units, tens, hundreds, thousands, conjunctions).
   - Convert normalized phrase to numeric value.
5. **Evaluation**
   - Metrics: field-detection IoU/F1, OCR token accuracy, end-to-end amount accuracy.
   - Analyze failure cases on noisy/complex checks.
6. **Packaging**
   - Provide reproducible notebooks/scripts for setup, training, inference, and evaluation.

## Recommended Tech Stack

1. Python 3.10+
2. OpenCV, Pillow, NumPy, Pandas, Matplotlib
3. Tesseract OCR + `pytesseract`
4. EasyOCR
5. TensorFlow/Keras or PyTorch for model experimentation
6. Scikit-learn for evaluation utilities

## Expected Output

Given a check image, the system should return:
1. extracted legal-amount text (Arabic),
2. normalized legal-amount phrase,
3. parsed numeric amount,
4. confidence/diagnostic info for downstream validation.

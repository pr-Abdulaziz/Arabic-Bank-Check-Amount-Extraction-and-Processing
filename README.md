# Arabic Bank Check Amount Extraction and Processing

This project focuses on extracting and verifying the **Arabic legal and courtesy amounts written on bank checks**. Based on the course document (`Term Project-ICS472.pdf`) and the project report (`Arabic_Check_Processing_Report.pdf`), the pipeline combines object detection, sequence recognition, Arabic amount conversion, and final legal/courtesy consistency checking.

## Project Idea

The target problem is challenging because Arabic bank-check text can include:
1. handwritten and printed variations,
2. noisy or low-quality scans,
3. segmented tokens and shape variations in Arabic script.

The proposed solution is a multi-stage pipeline:
1. detect the legal and courtesy amount regions on the full check image,
2. crop the detected regions,
3. recognize the courtesy amount as a digit sequence,
4. recognize the legal amount as Arabic text,
5. convert Arabic legal amount words into a numeric value,
6. compare the converted legal amount with the courtesy amount for verification.

## Current Repository Contents

1. `Images/`  
   Check images (`.tif`) used for experiments.
2. `BoundingBoxes/`  
   YOLO-format legal/courtesy amount annotations.
3. `CourtesyAmounts/`  
   Tokenized courtesy amount annotations.
4. `CourtesyAmounts_raw/`  
   Raw courtesy amount labels.
5. `LegalAmounts_raw_text/`  
   Raw Arabic legal amount labels.
6. `LegalAmounts_tokenized/`  
   Tokenized Arabic legal amount labels.
7. `ExampleAnnotations-BoundingBoxes.txt`  
   Sample bounding-box annotation in YOLO style (`class x_center y_center width height` normalized).
8. `ExampleAnnotations-LegalAmounts.txt`  
   Sample legal-amount annotation as tokenized Arabic words mapped to image names.
9. `Term Project-ICS472.pdf`  
   Course project brief and expected scope.
10. `Arabic_Check_Processing_Report.pdf`  
   Final report with methodology, implementation summary, and results.

## Project Notebooks

1. `project_notebook.ipynb`  
   Colab notebook containing the full project pipeline: Part A extraction, Part B courtesy recognition, Part C legal recognition, and Part D final verification.

## Quick Start

1. Upload the project folder to Google Drive as `Arabic-Bank-Check-Amount-Extraction-and-Processing`.
2. Open `project_notebook.ipynb` in Google Colab.
3. Run the setup cell and confirm that `Images/`, `BoundingBoxes/`, `CourtesyAmounts/`, `LegalAmounts_raw_text/`, and `LegalAmounts_tokenized/` are detected.
4. Run the notebook sections in order.

Part A writes:
1. `outputs/partA_output.txt`
2. `outputs/partA_metrics.txt`
3. `outputs/partA_annotation_audit.json`
4. `outputs/partA_split.json`

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
   - Metrics: IoU accuracy at 50%, 75%, and 90%, mean IoU, OCR token accuracy, CER/WER, and end-to-end verification accuracy.
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

For Part A, given a check image, the system should return:
1. check filename,
2. courtesy amount bounding box,
3. legal amount bounding box.

For the full project, the system should also return recognized courtesy digits, recognized legal text, converted legal numeric value, and final verification status.

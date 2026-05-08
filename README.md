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
4. `LegalAmounts/`
   Tokenized Arabic legal amount labels.
5. `CourtesyAmounts_raw/` and `LegalAmounts_raw_text/`
   Raw/reference labels kept for traceability only. They are not active notebook inputs.
6. `ExampleAnnotations-BoundingBoxes.txt`
   Sample bounding-box annotation in YOLO style (`class x_center y_center width height` normalized).
7. `ExampleAnnotations-LegalAmounts.txt`
   Sample legal-amount annotation as tokenized Arabic words mapped to image names.
8. `Term Project-ICS472.pdf`
   Course project brief and expected scope.
9. `Arabic_Check_Processing_Report.pdf`
   Final report with methodology, implementation summary, and results.

## Project Notebooks

1. `project_notebook.ipynb`  
   Colab notebook containing the full project pipeline: Part A extraction, Part B courtesy recognition, Part C legal recognition, Part D final verification, and final test-set evaluation.

## Quick Start

1. Upload the project folder to Google Drive as `Arabic-Bank-Check-Amount-Extraction-and-Processing`.
2. Open `project_notebook.ipynb` in Google Colab.
3. Run the setup cell and confirm that `Images/`, `BoundingBoxes/`, `CourtesyAmounts/`, and `LegalAmounts/` are detected.
4. Run the notebook sections in order.

Part A writes:
1. `results/partA_output.txt`
2. `results/partA_metrics.txt`
3. `results/partA_annotation_audit.json`
4. `results/partA_split.json`
5. `checkpoints/partA_yolov8n/best.pt`
6. `checkpoints/partA_yolov8n/last.pt`
7. `checkpoints/partA_yolov8n/model_info.json`

Part B writes:
1. `results/partB_output.txt`
2. `results/partB_metrics.txt`
3. `results/partB_crop_audit.json`
4. `checkpoints/partB_courtesy_crnn/best.pt`
5. `checkpoints/partB_courtesy_crnn/last.pt`
6. `checkpoints/partB_courtesy_crnn/model_info.json`

Part C writes:
1. `results/partC_output.txt`
2. `results/partC_metrics.txt`
3. `results/partC_crop_audit.json`
4. `checkpoints/partC_legal_crnn/best.pt`
5. `checkpoints/partC_legal_crnn/last.pt`
6. `checkpoints/partC_legal_crnn/model_info.json`

Part D writes:
1. `results/partD_output.txt`
2. `results/partD_metrics.txt`
3. `results/partD_details.json`

The notebook also saves report-ready artifacts under:
1. `results/tables/` for CSV metric tables, summary tables, training history, and error-analysis tables.
2. `results/json/` for machine-readable metric summaries.
3. `results/figures/` for histograms, loss curves, and final summary table images.
4. `results/error_analysis/` for confusion matrices, incorrect-sample examples, and ablation outputs.

When the instructor provides test data, place the test images in `CheckImages-Test/` and the optional test annotations in `CheckAnnotations-Test/`. The final test-set section in `project_notebook.ipynb` searches these folders recursively, so it also works if the uploaded folders contain one same-named nested folder.

Run the final test-set section in `project_notebook.ipynb` to generate:
1. `results/partA_test_output.txt`
2. `results/partA_test_metrics.txt`
3. `results/partB_test_output.txt`
4. `results/partB_test_metrics.txt`
5. `results/partC_test_output.txt`
6. `results/partC_test_metrics.txt`
7. `results/partD_test_output.txt`
8. `results/partD_test_metrics.txt`
9. `results/partD_test_details.json`

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
   - Metrics: IoU accuracy at 50%, 75%, and 90%, mean IoU, courtesy digit accuracy, CER, legal token/WER-style error rate, and end-to-end verification accuracy.
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

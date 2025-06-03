# License Plate OCR Comparison System

A web-based application for comparing and analyzing different preprocessing approaches in license plate OCR detection using EasyOCR.

## Features

- Multiple preprocessing approaches:

  - CLAHE + Sharpening + Gaussian + Otsu
  - Adaptive Threshold (Gaussian)
  - Basic Otsu Threshold
  - Sharpening + Otsu
  - Morphological Operations + Otsu

- Real-time comparison of two images
- Visualization of results including:

  - Confidence scores per approach
  - Character-wise accuracy
  - Prediction length comparison
  - Detailed results summary

- Support for Indonesian license plate formats
- Character correction for common OCR mistakes
- Noise word filtering
- Ground truth comparison

## Installation

1. Create a virtual environment:

```bash
python -m venv venvku
```

2. Activate the virtual environment:

```bash
.\venvku\Scripts\Activate.ps1
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. Upload two license plate images
4. (Optional) Enter ground truth for accuracy comparison
5. View the results and preprocessing visualizations

## Project Structure

```
ocr_web/
├── app.py              # Main Flask application
├── requirements.txt    # Project dependencies
├── static/
│   ├── css/           # Stylesheets
│   ├── uploads/       # Uploaded images
│   └── outputs/       # Processed images
└── templates/
    └── index.html     # Main web interface
```

## Features in Detail

### Preprocessing Approaches

1. **CLAHE + Sharpening + Gaussian + Otsu**

   - Contrast Limited Adaptive Histogram Equalization
   - Sharpening kernel application
   - Gaussian blur for noise reduction
   - Otsu thresholding

2. **Adaptive Threshold (Gaussian)**

   - Gaussian-weighted neighborhood analysis
   - Local adaptive thresholding

3. **Basic Otsu Threshold**

   - Standard Otsu's method
   - Global thresholding

4. **Sharpening + Otsu**

   - Image sharpening
   - Otsu thresholding

5. **Morphological Operations**
   - Dilation and Erosion
   - Opening and Closing
   - Otsu thresholding

### Post-processing

- Character correction mapping
- Indonesian license plate format validation
- Noise word filtering
- Geometric filtering of detected regions

## License Plate Format

Supports Indonesian license plate formats:

- 1-2 letters (Area code)
- 1-4 numbers (Registration number)
- 0-3 letters (Serial code)

Example: B1234ABC

## Development

Built with:

- Flask
- OpenCV
- EasyOCR
- NumPy
- Matplotlib
- Seaborn

## Notes

- GPU support is available through EasyOCR if CUDA is properly configured
- The system automatically detects dark/light backgrounds
- Preprocessing results are displayed for each approach
- Character-wise accuracy is calculated when ground truth is provided

## Requirements

- Python 3.8+
- CUDA (optional, for GPU support)
- See requirements.txt for full list of dependencies

from flask import Flask, render_template, request, url_for, jsonify
import os
import cv2
import numpy as np
import easyocr
import re
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import difflib
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize EasyOCR
reader = easyocr.Reader(["en"], gpu=False)

# --- Constants from original code ---
NOISE_WORDS_FOREIGN = [
    "YAMAHA",
    "HONDA",
    "SUZUKI",
    "KAWASAKI",
    "TOYOTA",
    "DAIHATSU",
    "MITSUBISHI",
    "NISSAN",
    "BMW",
    "MERCEDES",
    "AUDI",
    "HYUNDAI",
    "KIA",
    "WULING",
    "LEXUS",
    "SAMSAT",
    "DITLANTAS",
    "POLISI",
    "INDONESIA",
    "MOTOR",
    "MOBIL",
    "DEALER",
    "SERVICE",
    "EXPRESS",
    "AUTOMATIC",
    "MANUAL",
    "POWER",
    "STEERING",
    "INJECTION",
    "TURBO",
    "SPORT",
    "RACING",
    "TAHUN",
    "BULAN",
    "EXP",
    "EXPIRED",
    "PERPANJANG",
    "KADALUARSA",
    "VALID",
    "BERLAKU",
    "PLAT",
]
NOISE_WORDS_FILTERED = {word.upper() for word in NOISE_WORDS_FOREIGN if len(word) >= 3}

CHAR_CORRECTIONS = {
    "O": "0",
    "I": "1",
    "S": "5",
    "G": "6",
    "B": "8",
    "Z": "2",
    "A": "4",
    "o": "0",
    "i": "1",
    "s": "5",
    "g": "6",
    "b": "8",
    "z": "2",
    "a": "4",
    "T": "7",
    "L": "1",
    "D": "0",
    "E": "3",
    "0": "O",
    "1": "I",
    "5": "S",
    "6": "G",
    "8": "B",
    "2": "Z",
    "4": "A",
    "7": "T",
}


def is_valid_plate(text):
    """Validate Indonesian license plate format"""
    clean = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    patterns = [
        r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$",
    ]
    return (
        any(re.match(pattern, clean) for pattern in patterns) and 3 <= len(clean) <= 9
    )


def correct_characters_based_on_pattern(text, pattern_template):
    """Correct common OCR character misrecognitions based on pattern template"""
    corrected_text = list(text)

    for i, char_type in enumerate(pattern_template):
        if i >= len(corrected_text):
            break

        char = corrected_text[i]

        if char_type == "L":  # Expected to be a Letter
            if char.isdigit():
                if char in CHAR_CORRECTIONS and CHAR_CORRECTIONS[char].isalpha():
                    corrected_text[i] = CHAR_CORRECTIONS[char]
                else:
                    corrected_text[i] = char
            else:
                corrected_text[i] = char.upper()
        elif char_type == "N":  # Expected to be a Number
            if char.isalpha():
                if (
                    char.upper() in CHAR_CORRECTIONS
                    and CHAR_CORRECTIONS[char.upper()].isdigit()
                ):
                    corrected_text[i] = CHAR_CORRECTIONS[char.upper()]
                else:
                    corrected_text[i] = char

    return "".join(corrected_text)


def post_process_plate(text):
    """Advanced post-processing for license plate text"""
    if not text:
        return ""

    clean = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    if not clean:
        return ""

    match = re.match(r"^([A-Z]{0,2})(\d{1,4})([A-Z]{0,3})$", clean)

    if match:
        prefix_chars = match.group(1)
        number_chars = match.group(2)
        suffix_chars = match.group(3)

        corrected_prefix = correct_characters_based_on_pattern(prefix_chars, "LL")
        corrected_number = correct_characters_based_on_pattern(number_chars, "NNNN")
        corrected_suffix = correct_characters_based_on_pattern(suffix_chars, "LLL")

        processed = corrected_prefix + corrected_number + corrected_suffix
        return re.sub(r"[^A-Z0-9]", "", processed.upper())
    else:
        if not re.search(r"\d", clean):
            return ""
        corrected_fallback = "".join([CHAR_CORRECTIONS.get(c, c) for c in clean])
        return re.sub(r"[^A-Z0-9]", "", corrected_fallback.upper())


def char_accuracy(gt, pred):
    """Calculate character-wise accuracy"""
    gt_clean = re.sub(r"[^A-Z0-9]", "", str(gt).upper())
    pred_clean = re.sub(r"[^A-Z0-9]", "", str(pred).upper())
    if not gt_clean or not pred_clean:
        return 0.0
    matcher = difflib.SequenceMatcher(None, gt_clean, pred_clean)
    matches = sum(match.size for match in matcher.get_matching_blocks())
    return (
        matches / max(len(gt_clean), len(pred_clean))
        if max(len(gt_clean), len(pred_clean)) > 0
        else 0.0
    )


def filter_and_extract_main_plate_text(detected_results, min_char_height_ratio=0.5):
    """Filter dan gabungkan teks berdasarkan kriteria geometris"""
    boxes_info = []
    for bbox, text, conf in detected_results:
        tl, tr, br, bl = bbox
        height = max(bl[1], br[1]) - min(tl[1], tr[1])
        width = max(tr[0], br[0]) - min(tl[0], bl[0])
        text_alphanum = re.sub(r"[^A-Z0-9]", "", str(text).upper())

        if height < 8 or width < 8 or not text_alphanum:
            continue

        boxes_info.append(
            {
                "text": str(text),
                "conf": conf,
                "bbox": bbox,
                "text_alphanum": text_alphanum,
                "height": height,
                "width": width,
                "tl_x": tl[0],
                "center_y": (tl[1] + br[1]) / 2,
                "center_x": (tl[0] + br[0]) / 2,
            }
        )

    if not boxes_info:
        return "", 0.0

    boxes_info.sort(key=lambda b: b["height"], reverse=True)
    main_line_height_ref = boxes_info[0]["height"]
    main_line_y_ref = boxes_info[0]["center_y"]
    y_tolerance = main_line_height_ref * 0.4

    main_row_candidates = []
    for box in boxes_info:
        is_height_acceptable = (
            box["height"] >= main_line_height_ref * min_char_height_ratio
        )
        is_on_main_line = abs(box["center_y"] - main_line_y_ref) < y_tolerance

        if (
            box["text_alphanum"] in NOISE_WORDS_FILTERED
            and box["height"] < main_line_height_ref * 1.2
        ):
            continue

        if (box["text_alphanum"].isdigit() and len(box["text_alphanum"]) > 4) or (
            box["text_alphanum"].isalpha() and len(box["text_alphanum"]) > 3
        ):
            if box["height"] < main_line_height_ref * 0.7:
                continue

        if is_height_acceptable and is_on_main_line:
            main_row_candidates.append(box)

    if not main_row_candidates:
        boxes_info.sort(key=lambda b: b["height"], reverse=True)
        main_row_candidates = boxes_info[: min(len(boxes_info), 3)]

    if not main_row_candidates:
        return "", 0.0

    main_row_candidates.sort(key=lambda b: b["tl_x"])
    raw_combined_text = " ".join([b["text"] for b in main_row_candidates])
    final_text = " ".join(raw_combined_text.strip().split())
    final_conf = (
        np.mean([b["conf"] for b in main_row_candidates])
        if main_row_candidates
        else 0.0
    )

    return final_text, final_conf


def extract_potential_plates(text_blob):
    """Ekstrak kandidat plat nomor dari string teks"""
    if not text_blob:
        return []

    clean_text_for_regex = re.sub(r"[^A-Z0-9]", "", str(text_blob).upper())

    if not clean_text_for_regex:
        return []

    plate_pattern_regex = r"([A-Z]{1,2}\d{1,4}[A-Z]{0,3})"
    matches = list(re.finditer(plate_pattern_regex, clean_text_for_regex))

    if not matches:
        return []

    valid_length_matches = [m.group(0) for m in matches if 3 <= len(m.group(0)) <= 9]

    if not valid_length_matches:
        return []

    return [max(valid_length_matches, key=len)]


def enhanced_preprocessing(image):
    """Apply multiple preprocessing techniques with approach identification"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed_images = []
    approach_names = []

    # Define kernels
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Check if dark background
    h, w = gray.shape
    center_y_start, center_y_end = int(h * 0.25), int(h * 0.75)
    center_x_start, center_x_end = int(w * 0.25), int(w * 0.75)
    center_region = gray[center_y_start:center_y_end, center_x_start:center_x_end]
    mean_pixel_value = np.mean(center_region)
    is_dark_background = mean_pixel_value < 100

    # --- Approach 0: CLAHE + Sharpening + Gaussian + Otsu ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    sharpened = cv2.filter2D(cl1, -1, kernel_sharpening)
    blur_sharpened = cv2.GaussianBlur(sharpened, (3, 3), 0)
    if is_dark_background:
        _, thresh = cv2.threshold(
            blur_sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, thresh = cv2.threshold(
            blur_sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    preprocessed_images.append(thresh)
    approach_names.append("CLAHE+Sharpen+Gauss+Otsu")

    # --- Approach 1: Adaptive Threshold (Gaussian) ---
    sharpened_gray_adaptive = cv2.filter2D(gray, -1, kernel_sharpening)
    if is_dark_background:
        thresh_adaptive = cv2.adaptiveThreshold(
            sharpened_gray_adaptive,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
    else:
        thresh_adaptive = cv2.adaptiveThreshold(
            sharpened_gray_adaptive,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    preprocessed_images.append(thresh_adaptive)
    approach_names.append("Adaptive Threshold (Gaussian)")

    # --- Approach 2: Otsu Threshold ---
    if is_dark_background:
        _, thresh_otsu_base = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, thresh_otsu_base = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    preprocessed_images.append(thresh_otsu_base)
    approach_names.append("Otsu Threshold (Base)")

    # --- Approach 3: Sharpening + Otsu ---
    sharpened_gray_direct = cv2.filter2D(gray, -1, kernel_sharpening)
    if is_dark_background:
        _, thresh_sharpen_otsu = cv2.threshold(
            sharpened_gray_direct, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, thresh_sharpen_otsu = cv2.threshold(
            sharpened_gray_direct, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    preprocessed_images.append(thresh_sharpen_otsu)
    approach_names.append("Sharpening + Otsu")

    # --- Approach 4: Morphological Operations ---
    morphed_gray = gray.copy()
    morphed_gray = cv2.dilate(morphed_gray, kernel_small, iterations=1)
    morphed_gray = cv2.erode(morphed_gray, kernel_small, iterations=1)
    morphed_gray = cv2.morphologyEx(morphed_gray, cv2.MORPH_OPEN, kernel_medium)
    morphed_gray = cv2.morphologyEx(morphed_gray, cv2.MORPH_CLOSE, kernel_medium)
    if is_dark_background:
        _, thresh_all_morph_gray_otsu = cv2.threshold(
            morphed_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, thresh_all_morph_gray_otsu = cv2.threshold(
            morphed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    preprocessed_images.append(thresh_all_morph_gray_otsu)
    approach_names.append("Morph (Gray) + Otsu")

    return preprocessed_images, approach_names


def process_image_enhanced(image_path):
    """Enhanced image processing with preprocessing results"""
    image = cv2.imread(image_path)
    if image is None:
        return (None, "", "", "", 0.0, -1, "ERROR"), []

    preprocessed_images, approach_names = enhanced_preprocessing(image)
    preprocessing_results = []
    all_candidates_from_ocr = []

    for i, (prep_img, approach_name) in enumerate(
        zip(preprocessed_images, approach_names)
    ):
        try:
            # Get OCR results for this preprocessing approach
            ocr_detections = reader.readtext(prep_img, detail=1, paragraph=False)

            # Get filtered text and confidence
            geo_filtered_text, geo_combined_conf = filter_and_extract_main_plate_text(
                ocr_detections
            )

            if geo_filtered_text:
                all_candidates_from_ocr.append(
                    {
                        "geo_filtered_text": geo_filtered_text,
                        "conf": geo_combined_conf,
                        "approach": i,
                        "approach_name": approach_name,
                    }
                )

            # Draw bounding boxes on a copy of the original image
            detection_image = image.copy()
            for bbox, text, conf in ocr_detections:
                points = np.array(bbox).astype(np.int32)
                cv2.polylines(detection_image, [points], True, (0, 255, 0), 2)
                cv2.putText(
                    detection_image,
                    text,
                    tuple(points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # Convert images to base64
            _, prep_buffer = cv2.imencode(".png", prep_img)
            _, det_buffer = cv2.imencode(".png", detection_image)
            prep_base64 = base64.b64encode(prep_buffer).decode("utf-8")
            det_base64 = base64.b64encode(det_buffer).decode("utf-8")

            preprocessing_results.append(
                {
                    "approach_name": approach_name,
                    "preprocessed_image": prep_base64,
                    "detection_image": det_base64,
                }
            )

        except Exception as e:
            print(f"Error processing approach {approach_name}: {str(e)}")
            continue

    # Process kandidat terbaik seperti sebelumnya
    if not all_candidates_from_ocr:
        return (image, "", "", "", 0.0, -1, "NO_DETECTION"), preprocessing_results

    # Sort candidates by confidence
    all_candidates_from_ocr.sort(key=lambda x: x["conf"], reverse=True)

    best_final_candidate = {
        "final_processed_text": "",
        "conf": 0.0,
        "geo_filtered_text": "",
        "raw_extracted_text": "",
        "approach": -1,
        "approach_name": "NONE",
    }

    for candidate_info in all_candidates_from_ocr:
        text_from_geo_filter = candidate_info["geo_filtered_text"]
        conf_from_geo_filter = candidate_info["conf"]

        extracted_plate_candidates = extract_potential_plates(text_from_geo_filter)

        text_for_postprocessing = ""
        raw_extracted_text_for_df = text_from_geo_filter

        if extracted_plate_candidates:
            text_for_postprocessing = extracted_plate_candidates[0]
            raw_extracted_text_for_df = extracted_plate_candidates[0]
        else:
            text_for_postprocessing = text_from_geo_filter

        final_processed_text = post_process_plate(text_for_postprocessing)

        if (
            is_valid_plate(final_processed_text)
            and conf_from_geo_filter > best_final_candidate["conf"]
        ):
            best_final_candidate = {
                "final_processed_text": final_processed_text,
                "conf": conf_from_geo_filter,
                "geo_filtered_text": text_from_geo_filter,
                "raw_extracted_text": raw_extracted_text_for_df,
                "approach": candidate_info["approach"],
                "approach_name": candidate_info["approach_name"],
            }
        elif not best_final_candidate["final_processed_text"] and final_processed_text:
            best_final_candidate = {
                "final_processed_text": final_processed_text,
                "conf": conf_from_geo_filter,
                "geo_filtered_text": text_from_geo_filter,
                "raw_extracted_text": raw_extracted_text_for_df,
                "approach": candidate_info["approach"],
                "approach_name": candidate_info["approach_name"],
            }

    return (
        image,
        best_final_candidate["final_processed_text"],
        best_final_candidate["geo_filtered_text"],
        best_final_candidate["raw_extracted_text"],
        best_final_candidate["conf"],
        best_final_candidate["approach"],
        best_final_candidate["approach_name"],
    ), preprocessing_results


def create_comparison_visualization(
    results1, results2, ground_truth1="", ground_truth2=""
):
    """Create comparison visualization between two images"""
    plt.style.use("seaborn-v0_8")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Confidence comparison
    approaches = list(range(5))  # 5 approaches
    conf1 = [0] * 5
    conf2 = [0] * 5

    if results1 and len(results1) > 5:
        conf1[results1[5]] = results1[4]  # approach index and confidence
    if results2 and len(results2) > 5:
        conf2[results2[5]] = results2[4]

    x = np.arange(len(approaches))
    width = 0.35

    ax1.bar(x - width / 2, conf1, width, label="Image 1", alpha=0.8, color="skyblue")
    ax1.bar(x + width / 2, conf2, width, label="Image 2", alpha=0.8, color="lightcoral")
    ax1.set_xlabel("Preprocessing Approach")
    ax1.set_ylabel("Confidence Score")
    ax1.set_title("Confidence Score by Approach")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"App {i}" for i in approaches])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Character accuracy comparison (if ground truth provided)
    if ground_truth1 and ground_truth2 and results1 and results2:
        char_acc1 = char_accuracy(ground_truth1, results1[1]) * 100
        char_acc2 = char_accuracy(ground_truth2, results2[1]) * 100

        ax2.bar(
            ["Image 1", "Image 2"],
            [char_acc1, char_acc2],
            color=["skyblue", "lightcoral"],
            alpha=0.8,
        )
        ax2.set_ylabel("Character Accuracy (%)")
        ax2.set_title("Character-wise Accuracy Comparison")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate([char_acc1, char_acc2]):
            ax2.text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    else:
        ax2.text(
            0.5,
            0.5,
            "Ground Truth Required\nfor Accuracy Calculation",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Character Accuracy Comparison")

    # Prediction length comparison
    pred_len1 = len(results1[1]) if results1 and results1[1] else 0
    pred_len2 = len(results2[1]) if results2 and results2[1] else 0

    ax3.bar(
        ["Image 1", "Image 2"],
        [pred_len1, pred_len2],
        color=["skyblue", "lightcoral"],
        alpha=0.8,
    )
    ax3.set_ylabel("Prediction Length")
    ax3.set_title("Prediction Length Comparison")
    ax3.grid(True, alpha=0.3)

    # Results summary table
    ax4.axis("tight")
    ax4.axis("off")

    table_data = []
    table_data.append(["Metric", "Image 1", "Image 2"])
    table_data.append(
        [
            "Prediction",
            results1[1] if results1 else "N/A",
            results2[1] if results2 else "N/A",
        ]
    )
    table_data.append(
        [
            "Confidence",
            f"{results1[4]:.3f}" if results1 else "N/A",
            f"{results2[4]:.3f}" if results2 else "N/A",
        ]
    )
    table_data.append(
        [
            "Approach Used",
            results1[6] if results1 else "N/A",
            results2[6] if results2 else "N/A",
        ]
    )

    if ground_truth1 and ground_truth2:
        table_data.append(["Ground Truth", ground_truth1, ground_truth2])
        if results1 and results2:
            exact_match1 = "Yes" if results1[1] == ground_truth1 else "No"
            exact_match2 = "Yes" if results2[1] == ground_truth2 else "No"
            table_data.append(["Exact Match", exact_match1, exact_match2])

    table = ax4.table(cellText=table_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax4.set_title("Results Summary", pad=20, fontweight="bold")

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return plot_base64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
@app.route("/process", methods=["POST"])
def process_images():
    try:
        # Get uploaded files
        image1 = request.files.get("image1")
        image2 = request.files.get("image2")
        ground_truth1 = request.form.get("ground_truth1", "").strip()
        ground_truth2 = request.form.get("ground_truth2", "").strip()

        if not image1 or not image2:
            return jsonify({"error": "Both images are required"}), 400

        # Save uploaded files
        timestamp = str(int(time.time()))
        filename1 = secure_filename(f"{timestamp}_1_{image1.filename}")
        filename2 = secure_filename(f"{timestamp}_2_{image2.filename}")

        filepath1 = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
        filepath2 = os.path.join(app.config["UPLOAD_FOLDER"], filename2)

        image1.save(filepath1)
        image2.save(filepath2)

        # Process both images
        results1, preprocessing_results1 = process_image_enhanced(
            filepath1
        )  # Ubah untuk mengembalikan preprocessing_results
        results2, preprocessing_results2 = process_image_enhanced(filepath2)

        # Create comparison visualization
        comparison_plot = create_comparison_visualization(
            results1, results2, ground_truth1, ground_truth2
        )

        # Prepare response data
        response_data = {
            "success": True,
            "image1": {
                "filename": filename1,
                "prediction": results1[1] if results1 else "",
                "confidence": results1[4] if results1 else 0.0,
                "approach_used": results1[6] if results1 else "N/A",
                "raw_text": results1[2] if results1 else "",
                "ground_truth": ground_truth1,
                "preprocessing_results": preprocessing_results1,  # Tambahkan ini
            },
            "image2": {
                "filename": filename2,
                "prediction": results2[1] if results2 else "",
                "confidence": results2[4] if results2 else 0.0,
                "approach_used": results2[6] if results2 else "N/A",
                "raw_text": results2[2] if results2 else "",
                "ground_truth": ground_truth2,
                "preprocessing_results": preprocessing_results2,  # Tambahkan ini
            },
            "comparison_plot": comparison_plot,
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    try:
        # Get uploaded files
        image1 = request.files.get("image1")
        image2 = request.files.get("image2")
        ground_truth1 = request.form.get("ground_truth1", "").strip()
        ground_truth2 = request.form.get("ground_truth2", "").strip()

        if not image1 or not image2:
            return jsonify({"error": "Both images are required"}), 400

        # Save uploaded files
        timestamp = str(int(time.time()))
        filename1 = secure_filename(f"{timestamp}_1_{image1.filename}")
        filename2 = secure_filename(f"{timestamp}_2_{image2.filename}")

        filepath1 = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
        filepath2 = os.path.join(app.config["UPLOAD_FOLDER"], filename2)

        image1.save(filepath1)
        image2.save(filepath2)

        # Process both images
        results1 = process_image_enhanced(filepath1)
        results2 = process_image_enhanced(filepath2)

        # Create comparison visualization
        comparison_plot = create_comparison_visualization(
            results1, results2, ground_truth1, ground_truth2
        )

        # Prepare response data
        response_data = {
            "success": True,
            "image1": {
                "filename": filename1,
                "prediction": results1[1] if results1 else "",
                "confidence": results1[4] if results1 else 0.0,
                "approach_used": results1[6] if results1 else "N/A",
                "raw_text": results1[2] if results1 else "",
                "ground_truth": ground_truth1,
            },
            "image2": {
                "filename": filename2,
                "prediction": results2[1] if results2 else "",
                "confidence": results2[4] if results2 else 0.0,
                "approach_used": results2[6] if results2 else "N/A",
                "raw_text": results2[2] if results2 else "",
                "ground_truth": ground_truth2,
            },
            "comparison_plot": comparison_plot,
        }

        # Calculate additional metrics if ground truth is provided
        if ground_truth1 and results1:
            response_data["image1"]["char_accuracy"] = (
                char_accuracy(ground_truth1, results1[1]) * 100
            )
            response_data["image1"]["exact_match"] = (
                ground_truth1.upper() == results1[1].upper()
            )

        if ground_truth2 and results2:
            response_data["image2"]["char_accuracy"] = (
                char_accuracy(ground_truth2, results2[1]) * 100
            )
            response_data["image2"]["exact_match"] = (
                ground_truth2.upper() == results2[1].upper()
            )

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

"""
Dot-grid target uzerinden lens distortion olcumu yapar.

Yontem:
1. Her goruntude koyu grid noktalarini bulur.
2. Nokta merkezlerini yerel intensity-weighted centroid ile sub-pixel rafine eder.
3. Duzenli grid siralamasini cikarir.
4. Ideal planar grid -> goruntu homografisi kurar.
5. Homografi sonrasi kalan residual vektor alanini distortion olcumu olarak raporlar.

Bu, tek pozdaki nokta grid goruntuleri icin model-free bir olcumdur. Residual alan,
ideal bir duzlem projektif goruntusunden kalan sapmayi piksel ve mikrometre cinsinden
verir. Ek olarak OpenCV kalibrasyon modeliyle Brown/Conrady katsayilari fit edilebilir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png")


@dataclass
class DetectionResult:
    image_path: Path
    gray_raw: np.ndarray
    gray8: np.ndarray
    points: np.ndarray
    areas: np.ndarray
    rows: list[np.ndarray]
    row_lengths: list[int]
    ordered_points: np.ndarray | None
    metrics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noktali grid target ile lens distortion residual alanini olcer."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Grid goruntulerinin bulundugu klasor.",
    )
    parser.add_argument(
        "--glob",
        default="*.tiff",
        help="Goruntu dosya paterni. Ornek: '2x Lens v2_*.tiff'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/grid_distortion"),
        help="Cikti klasoru.",
    )
    parser.add_argument(
        "--grid-pitch-mm",
        type=float,
        default=0.125,
        help="Noktalar arasi hedef pitch, mm. ThorLabs grid icin 0.125 mm.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Kullanilacak grid satir sayisi. Bos birakilirsa otomatik secilir.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Kullanilacak grid sutun sayisi. Bos birakilirsa otomatik secilir.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Hizli deneme icin en fazla kac goruntu kullanilacagi.",
    )
    parser.add_argument(
        "--fit-opencv-model",
        action="store_true",
        help="OpenCV calibrateCamera ile Brown/Conrady katsayilari da fit et.",
    )
    parser.add_argument(
        "--visuals",
        type=int,
        default=3,
        help="Kac goruntu icin tespit/residual gorseli kaydedilecegi.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def list_images(input_dir: Path, pattern: str, max_images: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input klasoru bulunamadi: {input_dir}")

    images = [
        path
        for path in sorted(input_dir.glob(pattern), key=natural_key)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"Goruntu bulunamadi: {input_dir} / {pattern}")
    return images


def read_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Goruntu okunamadi: {image_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    low, high = np.percentile(gray_float, [0.5, 99.5])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((gray_float - low) * 255.0 / (high - low), 0, 255)
    return scaled.astype(np.uint8)


def refine_centers_weighted(
    gray: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """Koyu dairesel noktalar icin yerel agirlikli centroid rafinasyonu."""
    if len(centers) == 0:
        return centers

    gray_float = gray.astype(np.float32)
    height, width = gray.shape[:2]
    median_radius = max(4.0, math.sqrt(float(np.median(areas)) / math.pi))
    patch_radius = int(round(median_radius * 1.9))
    patch_radius = int(np.clip(patch_radius, 8, 30))
    refined = []

    for x_center, y_center in centers:
        x0 = max(0, int(round(x_center)) - patch_radius)
        y0 = max(0, int(round(y_center)) - patch_radius)
        x1 = min(width, int(round(x_center)) + patch_radius + 1)
        y1 = min(height, int(round(y_center)) + patch_radius + 1)
        patch = gray_float[y0:y1, x0:x1]
        if patch.size == 0:
            refined.append((x_center, y_center))
            continue

        background = float(np.percentile(patch, 90))
        weights = np.clip(background - patch, 0, None)
        total = float(weights.sum())
        if total <= 1e-6:
            refined.append((x_center, y_center))
            continue

        yy, xx = np.indices(patch.shape, dtype=np.float32)
        refined_x = float((weights * (xx + x0)).sum() / total)
        refined_y = float((weights * (yy + y0)).sum() / total)
        refined.append((refined_x, refined_y))

    return np.asarray(refined, dtype=np.float64)


def detect_dot_points(gray_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray8 = normalize_to_uint8(gray_raw)
    blurred = cv2.GaussianBlur(gray8, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        3,
    )

    num_labels, _labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    selected_centers = []
    selected_areas = []

    for label_index in range(1, num_labels):
        x, y, width, height, area = stats[label_index]
        if 80 <= area <= 1400 and 6 <= width <= 55 and 6 <= height <= 55:
            aspect = width / float(height)
            if 0.55 <= aspect <= 1.8:
                selected_centers.append(centers[label_index])
                selected_areas.append(area)

    points = np.asarray(selected_centers, dtype=np.float64)
    areas = np.asarray(selected_areas, dtype=np.float64)
    points = refine_centers_weighted(gray_raw, points, areas)
    return points, areas, gray8


def cluster_rows(points: np.ndarray) -> list[np.ndarray]:
    if len(points) == 0:
        return []

    points_y = points[np.argsort(points[:, 1])]
    y_values = points_y[:, 1]
    gaps = np.diff(y_values)
    large_gaps = gaps[gaps > 5.0]
    if len(large_gaps):
        threshold = max(5.0, float(np.median(large_gaps) * 0.45))
    else:
        threshold = max(5.0, float(np.percentile(gaps, 99)) * 0.5)

    rows = []
    start_index = 0
    for gap_index, gap in enumerate(gaps):
        if gap > threshold:
            row = points_y[start_index : gap_index + 1]
            rows.append(row[np.argsort(row[:, 0])])
            start_index = gap_index + 1

    row = points_y[start_index:]
    rows.append(row[np.argsort(row[:, 0])])
    return rows


def crop_ordered_grid(rows: list[np.ndarray], target_rows: int, target_cols: int) -> np.ndarray | None:
    if len(rows) < target_rows:
        return None

    all_points = np.vstack(rows)
    column_centers = cluster_axis_centers(all_points[:, 0])
    if len(column_centers) < target_cols:
        return None

    start_row = (len(rows) - target_rows) // 2
    selected_rows = rows[start_row : start_row + target_rows]
    start_col = (len(column_centers) - target_cols) // 2
    selected_columns = column_centers[start_col : start_col + target_cols]
    if len(selected_columns) < 2:
        return None

    column_pitch = float(np.median(np.diff(selected_columns)))
    max_assignment_distance = max(4.0, column_pitch * 0.35)
    ordered = []

    for row in selected_rows:
        if len(row) < target_cols:
            return None
        row_points = []
        used_indices = set()
        for column_center in selected_columns:
            distances = np.abs(row[:, 0] - column_center)
            nearest_index = int(np.argmin(distances))
            if nearest_index in used_indices or float(distances[nearest_index]) > max_assignment_distance:
                return None
            used_indices.add(nearest_index)
            row_points.append(row[nearest_index])
        ordered.append(np.asarray(row_points, dtype=np.float64))

    return np.vstack(ordered)


def cluster_axis_centers(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if len(sorted_values) == 0:
        return np.empty(0, dtype=np.float64)
    if len(sorted_values) == 1:
        return sorted_values

    gaps = np.diff(sorted_values)
    large_gaps = gaps[gaps > 5.0]
    threshold = max(5.0, float(np.median(large_gaps) * 0.45)) if len(large_gaps) else 5.0
    clusters = []
    start_index = 0
    for gap_index, gap in enumerate(gaps):
        if gap > threshold:
            clusters.append(sorted_values[start_index : gap_index + 1])
            start_index = gap_index + 1
    clusters.append(sorted_values[start_index:])
    return np.asarray([float(np.median(cluster)) for cluster in clusters], dtype=np.float64)


def local_dot_contrast(gray: np.ndarray, points: np.ndarray, areas: np.ndarray) -> float:
    if len(points) == 0 or len(areas) == 0:
        return float("nan")

    radius = max(3, int(round(math.sqrt(float(np.median(areas)) / math.pi))))
    inner_radius = max(2, int(round(radius * 0.9)))
    ring_inner = radius + 3
    ring_outer = radius + 7
    patch_radius = ring_outer
    yy, xx = np.ogrid[-patch_radius : patch_radius + 1, -patch_radius : patch_radius + 1]
    rr = xx * xx + yy * yy
    dot_mask = rr <= inner_radius * inner_radius
    ring_mask = (rr >= ring_inner * ring_inner) & (rr <= ring_outer * ring_outer)

    height, width = gray.shape[:2]
    values = []
    step = max(1, len(points) // 350)
    for x_center, y_center in points[::step]:
        x = int(round(x_center))
        y = int(round(y_center))
        if x - patch_radius < 0 or y - patch_radius < 0:
            continue
        if x + patch_radius >= width or y + patch_radius >= height:
            continue
        patch = gray[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
        dot_value = float(np.median(patch[dot_mask]))
        background_value = float(np.median(patch[ring_mask]))
        values.append(background_value - dot_value)

    return float(np.median(values)) if values else float("nan")


def image_metrics(gray_raw: np.ndarray, gray8: np.ndarray, points: np.ndarray, areas: np.ndarray) -> dict:
    raw = np.asarray(gray_raw)
    if np.issubdtype(raw.dtype, np.integer):
        saturation_value = np.iinfo(raw.dtype).max
    else:
        saturation_value = float(raw.max())

    percentiles = np.percentile(raw, [1, 5, 50, 95, 99])
    return {
        "detected_points": int(len(points)),
        "mean_intensity": float(raw.mean()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "max": float(raw.max()),
        "saturated_ratio": float(np.mean(raw >= saturation_value)),
        "laplacian_var": float(cv2.Laplacian(gray8, cv2.CV_64F).var()),
        "local_contrast_median": local_dot_contrast(raw, points, areas),
        "area_cv": float(np.std(areas) / np.mean(areas)) if len(areas) else float("nan"),
    }


def detect_image(image_path: Path) -> DetectionResult:
    gray_raw = read_grayscale_image(image_path)
    points, areas, gray8 = detect_dot_points(gray_raw)
    rows = cluster_rows(points)
    row_lengths = [len(row) for row in rows]
    metrics = image_metrics(gray_raw, gray8, points, areas)
    metrics.update(
        {
            "row_count": len(rows),
            "median_row_length": float(np.median(row_lengths)) if row_lengths else 0.0,
            "min_row_length": int(min(row_lengths)) if row_lengths else 0,
            "max_row_length": int(max(row_lengths)) if row_lengths else 0,
        }
    )

    return DetectionResult(
        image_path=image_path,
        gray_raw=gray_raw,
        gray8=gray8,
        points=points,
        areas=areas,
        rows=rows,
        row_lengths=row_lengths,
        ordered_points=None,
        metrics=metrics,
    )


def choose_grid_shape(
    detections: list[DetectionResult],
    rows_arg: int | None,
    cols_arg: int | None,
) -> tuple[int, int]:
    if rows_arg is not None and cols_arg is not None:
        return rows_arg, cols_arg

    row_counts = [len(result.rows) for result in detections if result.rows]
    central_row_lengths = []
    for result in detections:
        if not result.rows:
            continue
        row_count = len(result.rows)
        start = max(0, row_count // 2 - 5)
        central_row_lengths.extend(len(row) for row in result.rows[start : start + 10])

    if not row_counts or not central_row_lengths:
        raise RuntimeError("Grid satir/sutun sayisi otomatik belirlenemedi.")

    # Kenar satir/sutunlar bazi hedeflerde kismen kirpilabiliyor. Otomatik modda
    # tum goruntulerde kararlı kalacak merkezi ortak bolgeyi secmek daha saglam.
    auto_rows = int(math.floor(float(np.median(row_counts)) * 0.90))
    auto_cols = int(math.floor(float(np.median(central_row_lengths)) * 0.90))
    auto_rows = max(4, auto_rows)
    auto_cols = max(4, auto_cols)
    return rows_arg or auto_rows, cols_arg or auto_cols


def build_object_grid(rows: int, cols: int, pitch_mm: float) -> np.ndarray:
    object_points = np.zeros((rows * cols, 2), dtype=np.float32)
    object_points[:, :] = np.array(
        [[col * pitch_mm, row * pitch_mm] for row in range(rows) for col in range(cols)],
        dtype=np.float32,
    )
    return object_points


def homography_residuals(object_grid: np.ndarray, image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homography, _mask = cv2.findHomography(object_grid, image_points.astype(np.float32), 0)
    if homography is None:
        raise RuntimeError("Homografi hesaplanamadi.")

    predicted = cv2.perspectiveTransform(object_grid.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residual = image_points - predicted
    return homography, predicted, residual


def radial_summary(
    predicted_points: np.ndarray,
    residual_vectors: np.ndarray,
    image_size: tuple[int, int],
    bins: int = 10,
) -> list[dict]:
    width, height = image_size
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    vectors_from_center = predicted_points - center
    radius = np.linalg.norm(vectors_from_center, axis=1)
    unit = vectors_from_center / np.maximum(radius[:, None], 1e-9)
    radial = np.sum(residual_vectors * unit, axis=1)
    tangential = residual_vectors[:, 0] * unit[:, 1] - residual_vectors[:, 1] * unit[:, 0]

    max_radius = float(radius.max())
    rows = []
    for bin_index in range(bins):
        low = max_radius * bin_index / bins
        high = max_radius * (bin_index + 1) / bins
        mask = (radius >= low) & (radius < high if bin_index < bins - 1 else radius <= high)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": bin_index + 1,
                "radius_low_px": low,
                "radius_high_px": high,
                "points": int(np.sum(mask)),
                "radial_mean_px": float(np.mean(radial[mask])),
                "radial_abs_mean_px": float(np.mean(np.abs(radial[mask]))),
                "radial_p95_abs_px": float(np.percentile(np.abs(radial[mask]), 95)),
                "tangential_abs_mean_px": float(np.mean(np.abs(tangential[mask]))),
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_detection_visual(
    output_path: Path,
    gray8: np.ndarray,
    image_points: np.ndarray,
    predicted_points: np.ndarray,
    residual: np.ndarray,
) -> None:
    visual = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    for point in image_points:
        cv2.circle(visual, tuple(np.round(point).astype(int)), 2, (0, 255, 0), -1)

    magnification = 50.0
    for predicted, vector in zip(predicted_points[::4], residual[::4]):
        start = tuple(np.round(predicted).astype(int))
        end = tuple(np.round(predicted + vector * magnification).astype(int))
        cv2.arrowedLine(visual, start, end, (0, 0, 255), 1, tipLength=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visual)


def fit_opencv_model(
    object_grid_2d: np.ndarray,
    image_point_sets: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict:
    object_points_3d = np.zeros((len(object_grid_2d), 3), dtype=np.float32)
    object_points_3d[:, :2] = object_grid_2d
    object_points = [object_points_3d.copy() for _ in image_point_sets]
    image_points = [points.astype(np.float32).reshape(-1, 1, 2) for points in image_point_sets]

    rms, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion_coeffs)
        delta = img.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))

    return {
        "opencv_rms_px": float(rms),
        "opencv_reprojection_mean_px": float(np.mean(errors)),
        "opencv_reprojection_median_px": float(np.median(errors)),
        "opencv_reprojection_max_px": float(np.max(errors)),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion_coeffs.ravel().tolist(),
    }


def run_analysis(args: argparse.Namespace) -> dict:
    image_paths = list_images(args.input, args.glob, args.max_images)
    detections = [detect_image(path) for path in image_paths]
    target_rows, target_cols = choose_grid_shape(detections, args.rows, args.cols)
    object_grid = build_object_grid(target_rows, target_cols, args.grid_pitch_mm)

    valid_detections = []
    per_image_rows = []
    all_residuals = []
    all_predicted = []
    image_point_sets = []
    homographies = []

    for detection in detections:
        ordered = crop_ordered_grid(detection.rows, target_rows, target_cols)
        detection.ordered_points = ordered
        row = {
            "file": detection.image_path.name,
            "used": ordered is not None,
            **detection.metrics,
        }
        if ordered is None:
            per_image_rows.append(row)
            continue

        homography, predicted, residual = homography_residuals(object_grid, ordered)
        norms = np.linalg.norm(residual, axis=1)
        row.update(
            {
                "homography_rmse_px": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
                "residual_mean_abs_px": float(np.mean(norms)),
                "residual_p95_abs_px": float(np.percentile(norms, 95)),
                "residual_max_abs_px": float(np.max(norms)),
            }
        )
        per_image_rows.append(row)
        valid_detections.append(detection)
        all_residuals.append(residual)
        all_predicted.append(predicted)
        image_point_sets.append(ordered)
        homographies.append(homography)

    if not valid_detections:
        raise RuntimeError("Kullanilabilir grid goruntusu bulunamadi.")

    residual_stack = np.stack(all_residuals)
    predicted_stack = np.stack(all_predicted)
    median_predicted = np.median(predicted_stack, axis=0)
    median_residual = np.median(residual_stack, axis=0)
    residual_noise = residual_stack - median_residual[None, :, :]
    residual_norm = np.linalg.norm(median_residual, axis=1)
    noise_norm = np.linalg.norm(residual_noise, axis=2)

    image_height, image_width = valid_detections[0].gray_raw.shape[:2]
    pitch_px_x = np.diff(median_predicted.reshape(target_rows, target_cols, 2), axis=1)[:, :, 0]
    pitch_px_y = np.diff(median_predicted.reshape(target_rows, target_cols, 2), axis=0)[:, :, 1]
    pitch_px = float(np.median(np.concatenate([pitch_px_x.ravel(), pitch_px_y.ravel()])))
    micrometers_per_pixel = args.grid_pitch_mm * 1000.0 / pitch_px

    point_rows = []
    for index, (predicted, residual) in enumerate(zip(median_predicted, median_residual)):
        grid_row = index // target_cols
        grid_col = index % target_cols
        residual_px = float(np.linalg.norm(residual))
        point_rows.append(
            {
                "grid_row": grid_row,
                "grid_col": grid_col,
                "x_px": float(predicted[0]),
                "y_px": float(predicted[1]),
                "dx_px": float(residual[0]),
                "dy_px": float(residual[1]),
                "residual_px": residual_px,
                "dx_um": float(residual[0] * micrometers_per_pixel),
                "dy_um": float(residual[1] * micrometers_per_pixel),
                "residual_um": float(residual_px * micrometers_per_pixel),
                "repeatability_p95_px": float(np.percentile(noise_norm[:, index], 95)),
            }
        )

    radial_rows = radial_summary(
        median_predicted,
        median_residual,
        image_size=(image_width, image_height),
    )

    summary = {
        "input": str(args.input),
        "glob": args.glob,
        "images_total": len(image_paths),
        "images_used": len(valid_detections),
        "grid_rows": target_rows,
        "grid_cols": target_cols,
        "grid_points": target_rows * target_cols,
        "grid_pitch_mm": args.grid_pitch_mm,
        "image_width_px": image_width,
        "image_height_px": image_height,
        "median_pitch_px": pitch_px,
        "micrometers_per_pixel": micrometers_per_pixel,
        "homography_rmse_median_px": float(np.median([row["homography_rmse_px"] for row in per_image_rows if row["used"]])),
        "distortion_residual_mean_px": float(np.mean(residual_norm)),
        "distortion_residual_p95_px": float(np.percentile(residual_norm, 95)),
        "distortion_residual_max_px": float(np.max(residual_norm)),
        "distortion_residual_mean_um": float(np.mean(residual_norm) * micrometers_per_pixel),
        "distortion_residual_p95_um": float(np.percentile(residual_norm, 95) * micrometers_per_pixel),
        "distortion_residual_max_um": float(np.max(residual_norm) * micrometers_per_pixel),
        "repeatability_rms_px": float(np.sqrt(np.mean(residual_noise * residual_noise))),
        "repeatability_p95_px": float(np.percentile(noise_norm, 95)),
        "repeatability_max_px": float(np.max(noise_norm)),
        "notes": [
            "Residual, ideal planar gridin homografi ile aciklanamayan kismidir.",
            "Tek pozlu sabit hedefte bu alan distortion + hedef/kurulum kusurlari + tespit gurultusunu icerir.",
        ],
    }

    if args.fit_opencv_model:
        summary["opencv_model"] = fit_opencv_model(
            object_grid,
            image_point_sets,
            image_size=(image_width, image_height),
        )

    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "per_image_metrics.csv", per_image_rows)
    save_csv(args.output / "distortion_points.csv", point_rows)
    save_csv(args.output / "radial_summary.csv", radial_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for visual_index, detection in enumerate(valid_detections[: args.visuals]):
        homography, predicted, residual = homography_residuals(object_grid, detection.ordered_points)
        save_detection_visual(
            args.output / "visuals" / f"{detection.image_path.stem}_residual.png",
            detection.gray8,
            detection.ordered_points,
            predicted,
            residual,
        )

    return summary


def main() -> None:
    args = parse_args()
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
Olculen nokta-grid residual alanindan matematiksel distorsiyon modeli fit eder
ve goruntuleri bu modelle duzeltir.

Onemli not:
    Bu script ham goruntuden grid noktalarini yeniden tespit edip distorsiyonu
    olcmez. Distorsiyon olcumu daha once analyze_grid_distortion.py ile
    yapilir ve distortion_points.csv dosyasina yazilir.

Bu scriptin yaptigi is:
    1. distortion_points.csv icindeki olculmus x, y, dx, dy verilerini okur.
    2. Bu verilerden 5. derece 2B polinom distorsiyon modelini fit eder.
    3. Goruntudeki her piksel icin kaynak koordinat haritasi uretir.
    4. cv2.remap ile orijinal goruntuden okuyarak duzeltilmis goruntuyu olusturur.

Model:
    X = (x - cx) / scale
    Y = (y - cy) / scale

    dx(X, Y) = sum a_ij * X^i * Y^j
    dy(X, Y) = sum b_ij * X^i * Y^j

Burada (x, y) ideal/duzeltilmis piksel koordinatidir. cv2.remap icin kaynak
koordinati su sekilde hesaplanir:

    source_x = x + dx(X, Y)
    source_y = y + dy(X, Y)

Yani model, ideal piksel konumundan distorsiyonlu goruntudeki okunacak konuma
giden yer degistirme alanini verir.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid residual verisinden polinom distorsiyon modeli fit eder ve goruntuleri duzeltir."
    )
    parser.add_argument(
        "--points",
        type=Path,
        required=True,
        help="analyze_grid_distortion.py tarafindan uretilen distortion_points.csv.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Opsiyonel summary.json. Verilirse image size ve olcek buradan okunur.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Duzeltilecek goruntulerin bulundugu klasor.",
    )
    parser.add_argument(
        "--glob",
        default="*.tiff",
        help="Duzeltilecek goruntu deseni. Ornek: '3x Lens v2_*.tiff'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/distortion_corrected"),
        help="Duzeltilmis goruntulerin yazilacagi klasor.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Model JSON dosyasi. Bos birakilirsa output/distortion_model.json yazilir.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=5,
        help="2B polinom derecesi. Varsayilan: 5.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Test icin en fazla kac goruntu duzeltilecek.",
    )
    parser.add_argument(
        "--comparison-count",
        type=int,
        default=3,
        help="Kac goruntu icin original|corrected PNG karsilastirmasi kaydedilecegi.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def read_points(points_path: Path) -> dict[str, np.ndarray]:
    """Daha once olculmus grid residual verisini CSV'den okur.

    CSV icindeki x_px, y_px alanlari grid noktasinin goruntudeki konumudur.
    dx_px, dy_px alanlari ise bu noktanin ideal konumuna gore sapmasidir.

    Yani bu fonksiyon distorsiyonu olcmez; olculmus distorsiyon vektorlerini
    model fit islemi icin sayisal dizilere cevirir.
    """
    if not points_path.is_file():
        raise FileNotFoundError(f"Nokta CSV bulunamadi: {points_path}")

    rows = []
    with points_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Nokta CSV bos: {points_path}")

    return {
        "x": np.asarray([float(row["x_px"]) for row in rows], dtype=np.float64),
        "y": np.asarray([float(row["y_px"]) for row in rows], dtype=np.float64),
        "dx": np.asarray([float(row["dx_px"]) for row in rows], dtype=np.float64),
        "dy": np.asarray([float(row["dy_px"]) for row in rows], dtype=np.float64),
        "repeatability_p95": np.asarray(
            [float(row.get("repeatability_p95_px", 1.0)) for row in rows],
            dtype=np.float64,
        ),
    }


def polynomial_terms(degree: int) -> list[tuple[int, int]]:
    """i + j <= degree olacak sekilde 2B polinom terimlerini uretir.

    degree=5 icin ornek terimler:
        1, X, Y, X^2, X*Y, Y^2, ..., X^5, X^4*Y, ..., Y^5

    Bu terimler hem dx hem de dy fonksiyonu icin ayni sekilde kullanilir.
    """
    terms = []
    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            terms.append((x_power, y_power))
    return terms


def design_matrix(x_norm: np.ndarray, y_norm: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    """Polinom fit icin tasarim matrisini olusturur.

    Her satir bir grid noktasini, her sutun bir polinom terimini temsil eder.
    En kucuk kareler yontemi bu matrisi kullanarak a_ij ve b_ij katsayilarini
    hesaplar.
    """
    columns = []
    for x_power, y_power in terms:
        columns.append((x_norm**x_power) * (y_norm**y_power))
    return np.column_stack(columns)


def normalize_coordinates(x: np.ndarray, y: np.ndarray, cx: float, cy: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Piksel koordinatlarini goruntu merkezine gore yaklasik -1..+1 araligina tasir."""
    return (x - cx) / scale, (y - cy) / scale


def fit_polynomial_model(
    points: dict[str, np.ndarray],
    degree: int,
    image_width: int,
    image_height: int,
) -> dict:
    """Olculen residual alanindan polinom distorsiyon modelini fit eder.

    Girdi:
        points["x"], points["y"]  -> grid noktalarinin piksel konumlari
        points["dx"], points["dy"] -> bu noktalarda olculen sapmalar

    Cikti:
        d_x(X,Y) ve d_y(X,Y) fonksiyonlarini tanimlayan katsayilar.

    Burada bulunan katsayilar, daha sonra goruntudeki her piksel icin
    kaynak koordinati hesaplamada kullanilir.
    """
    cx = (image_width - 1) / 2.0
    cy = (image_height - 1) / 2.0
    scale = max(image_width, image_height) / 2.0
    terms = polynomial_terms(degree)

    # 1) Grid noktalarinin piksel koordinatlarini normalize et.
    x_norm, y_norm = normalize_coordinates(points["x"], points["y"], cx, cy, scale)

    # 2) Her grid noktasi icin polinom terimlerinden olusan tasarim matrisini kur.
    matrix = design_matrix(x_norm, y_norm, terms)

    # Daha dusuk tekrar edilebilirlik hatasi olan noktalar fit icinde biraz daha guclu olsun.
    weights = 1.0 / np.maximum(points["repeatability_p95"], 0.03)
    weights = weights / np.median(weights)
    weighted_matrix = matrix * weights[:, None]

    # 3) En kucuk kareler ile dx ve dy icin ayri katsayi setleri hesapla.
    #    coeff_dx -> d_x(X,Y) fonksiyonunun a_ij katsayilari
    #    coeff_dy -> d_y(X,Y) fonksiyonunun b_ij katsayilari
    coeff_dx, *_ = np.linalg.lstsq(weighted_matrix, points["dx"] * weights, rcond=None)
    coeff_dy, *_ = np.linalg.lstsq(weighted_matrix, points["dy"] * weights, rcond=None)

    # 4) Modelin olculen residual alanini ne kadar iyi temsil ettigini hesapla.
    pred_dx = matrix @ coeff_dx
    pred_dy = matrix @ coeff_dy
    err_x = points["dx"] - pred_dx
    err_y = points["dy"] - pred_dy
    err_norm = np.sqrt(err_x * err_x + err_y * err_y)

    return {
        "model_type": "2d_polynomial_displacement",
        "degree": degree,
        "image_width_px": image_width,
        "image_height_px": image_height,
        "cx_px": cx,
        "cy_px": cy,
        "normalization_scale_px": scale,
        "terms": [{"x_power": i, "y_power": j} for i, j in terms],
        "coeff_dx_px": coeff_dx.tolist(),
        "coeff_dy_px": coeff_dy.tolist(),
        "fit_metrics": {
            "points": int(len(points["x"])),
            "rmse_px": float(np.sqrt(np.mean(err_norm * err_norm))),
            "mean_abs_px": float(np.mean(err_norm)),
            "p95_abs_px": float(np.percentile(err_norm, 95)),
            "max_abs_px": float(np.max(err_norm)),
            "dx_rmse_px": float(np.sqrt(np.mean(err_x * err_x))),
            "dy_rmse_px": float(np.sqrt(np.mean(err_y * err_y))),
        },
        "equation": {
            "coordinate_normalization": "X=(x-cx_px)/normalization_scale_px, Y=(y-cy_px)/normalization_scale_px",
            "dx": "sum(coeff_dx_px[k] * X^terms[k].x_power * Y^terms[k].y_power)",
            "dy": "sum(coeff_dy_px[k] * X^terms[k].x_power * Y^terms[k].y_power)",
            "remap": "corrected(x,y) = distorted(x + dx(x,y), y + dy(x,y))",
        },
    }


def evaluate_model(model: dict, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Verilen piksel koordinatlari icin modelden dx, dy sapmalarini hesaplar.

    Bu fonksiyon tek bir nokta icin de, tum goruntu piksel izgara matrisi icin de
    calisabilir. build_remap icinde tum 2048 x 2048 piksel icin topluca kullanilir.
    """
    terms = [(term["x_power"], term["y_power"]) for term in model["terms"]]
    x_norm, y_norm = normalize_coordinates(
        x,
        y,
        float(model["cx_px"]),
        float(model["cy_px"]),
        float(model["normalization_scale_px"]),
    )
    matrix = design_matrix(x_norm.ravel(), y_norm.ravel(), terms)
    dx = matrix @ np.asarray(model["coeff_dx_px"], dtype=np.float64)
    dy = matrix @ np.asarray(model["coeff_dy_px"], dtype=np.float64)
    return dx.reshape(x.shape), dy.reshape(y.shape)


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
        raise FileNotFoundError(f"Duzeltilecek goruntu bulunamadi: {input_dir} / {pattern}")
    return images


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Goruntu okunamadi: {image_path}")
    return image


def build_remap(model: dict) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap icin kaynak koordinat haritalarini olusturur.

    Bu fonksiyon modelin goruntuye uygulandigi ana yerdir.

    Her cikti pikseli (x, y) icin:
        1. d_x(x,y), d_y(x,y) hesaplanir.
        2. source_x = x + d_x(x,y)
           source_y = y + d_y(x,y)
        3. map_x ve map_y icine bu kaynak koordinatlar yazilir.

    Sonra cv2.remap, duzeltilmis goruntudeki her pikseli doldurmak icin
    orijinal goruntude map_x/map_y ile gosterilen koordinattan okuma yapar.
    """
    width = int(model["image_width_px"])
    height = int(model["image_height_px"])

    # Cikti goruntusundeki tum piksel koordinatlarini olustur.
    # grid_x[y, x] = x koordinati, grid_y[y, x] = y koordinati.
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    # Her piksel icin polinom modelden sapma miktarini hesapla.
    dx, dy = evaluate_model(model, grid_x.astype(np.float64), grid_y.astype(np.float64))

    # Ters haritalama:
    # Duzeltilmis goruntudeki (x,y) pikseli, orijinal goruntude
    # (x+dx, y+dy) konumundan okunur.
    map_x = (grid_x + dx.astype(np.float32)).astype(np.float32)
    map_y = (grid_y + dy.astype(np.float32)).astype(np.float32)
    return map_x, map_y


def correct_image(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Orijinal goruntuyu kaynak koordinat haritalariyla duzeltir.

    cv2.remap her cikti pikseli icin map_x/map_y koordinatlarindan deger okur.
    Koordinat tam sayi degilse INTER_CUBIC interpolasyonla ara piksel degeri
    hesaplanir.
    """
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize_preview(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.dtype == np.uint8:
        return gray
    low, high = np.percentile(gray.astype(np.float32), [0.5, 99.5])
    if high <= low:
        high = low + 1.0
    return np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def save_comparison(original: np.ndarray, corrected: np.ndarray, output_path: Path) -> None:
    original8 = normalize_preview(original)
    corrected8 = normalize_preview(corrected)
    comparison = np.hstack([original8, corrected8])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), comparison)


def load_summary(summary_path: Path | None) -> dict:
    if summary_path is None:
        return {}
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary bulunamadi: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_model_report(model: dict, output_path: Path) -> None:
    lines = []
    lines.append("Polinom Distorsiyon Modeli")
    lines.append("==========================")
    lines.append("")
    lines.append("Normalize koordinatlar:")
    lines.append("X = (x - cx) / scale")
    lines.append("Y = (y - cy) / scale")
    lines.append("")
    lines.append(f"cx = {model['cx_px']:.6f} px")
    lines.append(f"cy = {model['cy_px']:.6f} px")
    lines.append(f"scale = {model['normalization_scale_px']:.6f} px")
    lines.append(f"degree = {model['degree']}")
    lines.append("")
    lines.append("Duzeltme modeli:")
    lines.append("source_x = x + dx(X,Y)")
    lines.append("source_y = y + dy(X,Y)")
    lines.append("corrected(x,y) = distorted(source_x, source_y)")
    lines.append("")
    lines.append("Fit metrikleri:")
    for key, value in model["fit_metrics"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Terimler ve katsayilar:")
    lines.append("x_power,y_power,coeff_dx_px,coeff_dy_px")
    for term, coeff_x, coeff_y in zip(model["terms"], model["coeff_dx_px"], model["coeff_dy_px"]):
        lines.append(f"{term['x_power']},{term['y_power']},{coeff_x:.12g},{coeff_y:.12g}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    # Asama 0:
    # Bu scriptin bekledigi olcum girdileri:
    #   --points  -> analyze_grid_distortion.py tarafindan uretilen distortion_points.csv
    #   --summary -> goruntu boyutu ve olcek gibi ek bilgiler
    #   --input   -> duzeltilecek orijinal TIFF goruntu klasoru
    summary = load_summary(args.summary)

    # Asama 1:
    # Daha once olculmus grid residual vektorlerini oku.
    # Burada ham goruntu analizi yapilmiyor; dx/dy sapmalari CSV'den geliyor.
    points = read_points(args.points)
    image_width = int(summary.get("image_width_px", 2048))
    image_height = int(summary.get("image_height_px", 2048))

    # Asama 2:
    # Olculmus residual alanindan matematiksel polinom modeli kur.
    # Bu adim a_ij ve b_ij katsayilarini hesaplar.
    model = fit_polynomial_model(points, args.degree, image_width, image_height)
    if summary:
        model["source_summary"] = summary

    # Asama 3:
    # Kurulan modeli daha sonra tekrar kullanabilmek ve raporlayabilmek icin kaydet.
    args.output.mkdir(parents=True, exist_ok=True)
    model_output = args.model_output or (args.output / "distortion_model.json")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    write_model_report(model, args.output / "distortion_model_equation.txt")

    print(json.dumps(model["fit_metrics"], indent=2))
    print(f"Model yazildi: {model_output}")

    if args.input is None:
        return

    # Asama 4:
    # Polinom modeli goruntudeki her piksel icin uygula ve cv2.remap'in
    # kullanacagi kaynak koordinat haritalarini olustur.
    map_x, map_y = build_remap(model)

    # Asama 5:
    # Duzeltilecek orijinal goruntuleri listele.
    image_paths = list_images(args.input, args.glob, args.max_images)
    corrected_dir = args.output / "corrected_images"
    comparison_dir = args.output / "comparisons"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Asama 6:
    # Her goruntu icin:
    #   - goruntuyu oku
    #   - map_x/map_y ile ters haritalama uygula
    #   - duzeltilmis goruntuyu kaydet
    #   - ilk birkac goruntu icin original|corrected karsilastirmasi kaydet
    for index, image_path in enumerate(image_paths):
        image = read_image(image_path)
        corrected = correct_image(image, map_x, map_y)
        output_path = corrected_dir / image_path.name
        cv2.imwrite(str(output_path), corrected)

        if index < args.comparison_count:
            comparison_path = comparison_dir / f"{image_path.stem}_original_vs_corrected.png"
            save_comparison(image, corrected, comparison_path)

    print(f"Duzeltilen goruntu sayisi: {len(image_paths)}")
    print(f"Cikti klasoru: {corrected_dir}")


if __name__ == "__main__":
    main()

"""
Distorsiyon duzeltme isleminin calisma suresini olcer.

Bu benchmark, apply_distortion_model.py icindeki ayni model uygulama yolunu kullanir:
    1. distortion_model.json dosyasini yukler.
    2. map_x / map_y ters haritalama tablolarini olusturur.
    3. Orijinal goruntuye cv2.remap uygular.
    4. Okuma, remap ve yazma surelerini ayri ayri raporlar.

Not:
    Bu Python/OpenCV benchmark'i calistigi bilgisayarin performansini olcer.
    PolarFire RISC-V uzerinde kesin sure icin ayni mantigin C/C++ tarafinda
    readmcycle() ile olculmesi gerekir.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2

from apply_distortion_model import build_remap, correct_image, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distorsiyon duzeltme runtime benchmark.")
    parser.add_argument("--model", type=Path, required=True, help="distortion_model.json yolu.")
    parser.add_argument("--image", type=Path, required=True, help="Duzeltilecek tek test goruntusu.")
    parser.add_argument("--output", type=Path, default=Path("runtime_benchmark_output.tiff"))
    parser.add_argument("--iterations", type=int, default=25, help="Remap tekrar sayisi.")
    parser.add_argument("--threads", type=int, default=1, help="OpenCV thread sayisi.")
    parser.add_argument("--json-output", type=Path, default=None, help="Opsiyonel JSON rapor yolu.")
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percent / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(args.threads)

    model = json.loads(args.model.read_text(encoding="utf-8"))

    # Modelden map_x/map_y haritalari bir kez olusturulur.
    # Gercek sistemde bu adim acilista/precompute olarak yapilabilir.
    t0 = time.perf_counter()
    map_x, map_y = build_remap(model)
    map_build_s = time.perf_counter() - t0

    # Tek test goruntusunu diskten oku.
    t0 = time.perf_counter()
    image = read_image(args.image)
    read_s = time.perf_counter() - t0

    # Ilk calistirma cache/warm-up etkisi icin olcum disinda birakilir.
    corrected = correct_image(image, map_x, map_y)

    remap_times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        corrected = correct_image(image, map_x, map_y)
        remap_times.append(time.perf_counter() - t0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cv2.imwrite(str(args.output), corrected)
    write_s = time.perf_counter() - t0

    pixels = int(image.shape[0] * image.shape[1])
    remap_median_s = statistics.median(remap_times)

    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "image_path": str(args.image),
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "pixels": pixels,
        "iterations": args.iterations,
        "map_build_once_s": map_build_s,
        "read_one_s": read_s,
        "remap_only_median_s": remap_median_s,
        "remap_only_mean_s": statistics.mean(remap_times),
        "remap_only_p95_s": percentile(remap_times, 95),
        "write_one_s": write_s,
        "end_to_end_cached_map_median_s": read_s + remap_median_s + write_s,
        "ns_per_pixel_remap": remap_median_s / pixels * 1e9,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

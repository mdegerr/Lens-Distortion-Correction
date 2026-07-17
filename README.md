# Lens Distortion Measurement and Correction

Bu proje, Mitutoyo CF 2X, 3X ve 5X lenslerle alınan ThorLabs 0.125 mm pitch noktalı grid görüntülerinden geometrik distorsiyon alanını ölçmek, her lens için matematiksel düzeltme modeli oluşturmak ve görüntüleri ters haritalama yöntemiyle düzeltmek için hazırlanmıştır.

Ham TIFF görüntülerin tamamı GitHub deposuna dahil edilmemiştir. Depoda yalnızca çalıştırılabilir analiz kodları, küçük model/özet JSON dosyaları, rapor notları ve her lensin v2 veri setinden seçilmiş küçük bir örnek görüntü alt kümesi tutulur.

## Projenin Amacı

- Grid target üzerindeki nokta merkezlerini tespit etmek.
- Homografi ile ideal grid konumlarını hesaplamak.
- Ölçülen nokta ile ideal nokta arasındaki residual vektörlerden distorsiyon alanını çıkarmak.
- 2X, 3X ve 5X lensler için ayrı 5. derece 2B polinom düzeltme modeli oluşturmak.
- Hazır model ile görüntüleri `cv2.remap` tabanlı ters haritalama yöntemiyle düzeltmek.
- Düzeltme öncesi ve sonrası hata metriklerini karşılaştırmak.

## Temel Matematiksel Model

Piksel koordinatları önce görüntü merkezine göre normalize edilir:

```text
X = (x - cx) / s
Y = (y - cy) / s
```

Her lens için x ve y yönündeki sapma ayrı polinomlarla modellenir:

```text
d_x(x,y) = Σ_{i+j≤5} a_ij · X^i · Y^j
d_y(x,y) = Σ_{i+j≤5} b_ij · X^i · Y^j
```

Düzeltilmiş görüntü ters haritalama ile oluşturulur:

```text
I_corrected(x,y) = I_distorted(x + d_x(x,y), y + d_y(x,y))
```

Bu denklem, düzeltilmiş görüntüdeki her `(x,y)` pikseli için orijinal distorsiyonlu görüntüde hangi koordinattan değer okunacağını hesaplar.

## Klasör Yapısı

```text
lens-distortion-correction/
  README.md
  requirements.txt
  .gitignore
  data/
    README.md
    sample/
      2x/
      3x/
      5x/
  docs/
    Distorsiyon_Modelleme_Raporu.md
  results/
    2x/
      summary.json
      distortion_model.json
    3x/
      summary.json
      distortion_model.json
    5x/
      summary.json
      distortion_model.json
    lens_distortion_type_analysis.json
    runtime_benchmark/
      3x_runtime_benchmark.json
  scripts/
    analyze_grid_distortion.py
    apply_distortion_model.py
    benchmark_distortion_runtime.py
```

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Kullanım

### 1. Grid Distorsiyon Ölçümü

Ham görüntüler `data/` altında tutulabilir. Örnek komut:

```powershell
python scripts\analyze_grid_distortion.py `
  --input data\3x `
  --glob "3x Lens v2_*.tiff" `
  --output output\3x_v2_distortion `
  --grid-pitch-mm 0.125 `
  --rows 59 `
  --cols 59
```

Bu adım `summary.json`, `distortion_points.csv`, `radial_summary.csv` ve görsel doğrulama çıktıları üretir.

### 2. Matematiksel Model ve Görüntü Düzeltme

```powershell
python scripts\apply_distortion_model.py `
  --points output\3x_v2_distortion\distortion_points.csv `
  --summary output\3x_v2_distortion\summary.json `
  --input data\3x `
  --glob "3x Lens v2_*.tiff" `
  --output output\3x_v2_correction_model `
  --degree 5
```

Bu adım `distortion_model.json` dosyasını oluşturur ve düzeltilmiş görüntüleri `corrected_images/` klasörüne yazar.

### 3. Runtime Benchmark

```powershell
python scripts\benchmark_distortion_runtime.py `
  --model results\3x\distortion_model.json `
  --image data\3x\ornek.tiff `
  --output output\benchmark\3x_corrected.tiff `
  --json-output output\benchmark\3x_runtime.json `
  --iterations 25 `
  --threads 1
```

PC üzerinde yapılan referans ölçümde 2048 x 2048 piksel 16-bit tek görüntü için sadece remapping süresi yaklaşık 85 ms ölçülmüştür. Hedef PolarFire donanımı mevcut olmadığından donanım üzerindeki kesin süre ölçülmemiştir; nihai değer `readmcycle()` ile doğrulanmalıdır.

## Özet Sonuçlar

| Lens | Veri seti | Grid noktası | P95 hata önce | P95 hata sonra | Distorsiyon tipi yorumu |
|---|---|---:|---:|---:|---|
| 2X | v2 | 2116 | 0.7812 px | 0.6626 px | Karışık / asimetrik |
| 3X | v2 | 3481 | 0.4708 px | 0.1535 px | Karışık / mustache benzeri |
| 5X | v2 | 1681 | 0.6603 px | 0.2541 px | Karışık / mustache benzeri |

## Veri Politikası

Tam ham `.tif/.tiff` veri setleri, sanal ortamlar ve büyük çıktı dosyaları depoya eklenmez. Bu dosyalar yerel olarak veya ayrı bir veri paylaşım alanında tutulmalıdır.

GitHub kotasını doldurmamak için yalnızca küçük bir örnek veri alt kümesi eklenmiştir:

| Lens | Veri seti | Eklenen örnek görüntü sayısı |
|---|---|---:|
| 2X | v2 | 5 |
| 3X | v2 | 5 |
| 5X | v2 | 5 |

Tam ölçüm sonuçları 100 görüntülük v2 setleriyle üretilmiştir; `data/sample/` klasörü yalnızca kodun çalışma biçimini göstermek ve küçük ölçekli deneme yapmak içindir.

## Notlar

- Model katsayıları aynı lens, aynı kamera geometrisi, aynı çalışma mesafesi ve aynı çözünürlük için geçerlidir.
- Optik kurulum değişirse model yeniden çıkarılmalıdır.
- `results/` klasöründeki JSON dosyaları küçük ve izlenebilir özet/model çıktılarıdır.

# Mitutoyo CF 2X, 3X ve 5X Lenslerde Grid Tabanlı Distorsiyon Ölçümü ve Matematiksel Düzeltme Modeli

## Özet
Bu çalışmada Mitutoyo CF 2X, 3X ve 5X lensler ile elde edilen ThorLabs 0.125 mm pitch noktalı grid görüntülerinden geometrik distorsiyon ölçülmüş, her lens için ayrı matematiksel düzeltme modeli kurulmuş ve modelin düzeltme başarısı sayısal olarak doğrulanmıştır.

**Anahtar kelimeler:** lens distorsiyonu, grid target, homografi, polinom model, ters haritalama, OpenCV

## İçindekiler

1. Giriş
2. Teorik Arka Plan
3. Materyal ve Yöntem
4. Matematiksel Model
5. Bulgular
6. 2X Lens Ek İncelemesi
7. Tartışma ve Sonuç
8. Kaynakça

## 1. Giriş
Optik ölçüm sistemlerinde lens distorsiyonu, görüntüdeki koordinatların ideal geometriden sapmasına neden olur. Mikrometre seviyesinde ölçüm hedeflenen sistemlerde bu sapmanın ölçülmesi ve düzeltilmesi görüntüden elde edilen fiziksel sonuçların güvenilirliği için kritiktir.

## 2. Teorik Arka Plan
Literatürde lens distorsiyonu ideal pinhole kamera modelinden sapma olarak tanımlanır. Brown-Conrady ve OpenCV modelleri radyal ve teğetsel bozulma terimlerini kullanır. Bu çalışmada aynı fiziksel sapma, ölçülen grid residual alanından ampirik iki boyutlu polinom model olarak elde edilmiştir.

## 3. Materyal ve Yöntem
Her görüntüde grid nokta merkezleri tespit edildi. Fiziksel olarak düzgün aralıklı gridin görüntü düzlemindeki ideal konumu homografi ile modellendi. Ölçülen nokta merkezleri ile homografi tarafından açıklanan ideal noktalar arasındaki fark, distorsiyon vektörü olarak kabul edildi. Bu vektör alanı 2B polinom modelle temsil edildi ve cv2.remap mantığıyla ters haritalama olarak görüntüye uygulandı.

## 4. Matematiksel Model

İdeal grid noktası homografi ile görüntü düzlemine taşınır:

`p_ideal ~ H P_grid`

Ölçülen distorsiyon vektörü şu şekilde tanımlanır:

`r = p_observed - p_ideal = [dx, dy]^T`

Her lens için 5. derece 2B polinom yer değiştirme modeli fit edilmiştir:

`dx(X,Y)=sum(a_ij X^i Y^j), dy(X,Y)=sum(b_ij X^i Y^j)`

Düzeltme ters haritalama ile uygulanmıştır:

`I_corrected(x,y)=I_distorted(x+dx(x,y), y+dy(x,y))`

Bu denklemler, düzeltilmiş görüntüdeki her pikselin orijinal distorsiyonlu görüntüde hangi koordinattan okunacağını belirler.

## 5. Bulgular

| Lens | Görüntü | Grid | Önce P95 px | Sonra P95 px | İyileşme | Önce P95 µm | Sonra P95 µm | Tip yorumu |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2X | 100 | 2116 | 0.7812 | 0.6626 | %15.2 | 2.1785 | 1.8476 | Karışık / mustache benzeri |
| 3X | 100 | 3481 | 0.4708 | 0.1535 | %67.4 | 1.8980 | 0.6189 | Karışık / mustache benzeri |
| 5X | 100 | 1681 | 0.6603 | 0.2541 | %61.5 | 1.6474 | 0.6341 | Karışık / mustache benzeri |

Bu tabloya göre 3X ve 5X lenslerde düzeltme etkisi belirgindir. 2X lens için iyileşme daha sınırlıdır; bunun nedeni residual alanın tek yönlü ve düzenli bir radyal distorsiyon karakteri göstermemesidir.

### Grafiklerin Yorumu

- P95 önce/sonra grafiği, modelin pratik düzeltme etkisini gösterir. Sütunlar arasındaki fark büyüdükçe modelin etkisi artar.
- Model fit RMSE grafiği, polinom modelin ölçülen residual alanı ne kadar iyi temsil ettiğini gösterir. Düşük RMSE daha güvenilir model anlamına gelir.
- Radyal profil grafiği, barrel/pincushion/mustache karakterini yorumlamak için kullanılır. Sıfır çizgisinin altında merkeze doğru, üstünde dışa doğru sapma vardır.

## 4. 2X Lens Düzeltmesinin Sınırlı Kalmasının Yorumu

2X lens için düzeltme etkisi 3X ve 5X'e göre daha sınırlıdır. Adil kıyas için ilk 5 orijinal görüntü ile ilk 5 düzeltilmiş görüntü ayrıca karşılaştırılmıştır.

| Kriter | Değer | Yorum |
|---|---:|---|
| İlk 5 görüntü önce P95 | 0.8053 px | Düzeltme öncesi aynı görüntü grubunun hatası |
| İlk 5 görüntü sonra P95 | 0.6626 px | Düzeltme sonrası kalan hata |
| İyileşme | %17.7 | 2X için sınırlı fakat ölçülebilir azalma |
| Model fit RMSE | 0.2718 px | Polinom modelin residual alanı açıklama hatası |
| Pozitif / negatif radyal oran | %53.6 / %46.4 | Radyal karakter tek yönlü değil |
| Teğetsel/toplam oran | 0.548 | Asimetrik/teğetsel bileşen yüksek |

Bu bulgular, 2X veri setinde hatanın yalnızca düzenli barrel veya pincushion distorsiyondan oluşmadığını; lokal, asimetrik veya kurulum/tespit kaynaklı bileşenlerin daha baskın olduğunu gösterir.

## 7. Tartışma ve Sonuç

Ölçüme dayalı polinom model yaklaşımı 3X ve 5X lenslerde güçlü düzeltme sağlamıştır. 2X lens için ise teğetsel/asimetrik bileşen yüksek olduğundan düzeltme etkisi sınırlı kalmıştır. Bu sonuç, model başarısının yalnızca model derecesine değil, residual alanın sistematik ve tekrar edilebilir olmasına da bağlı olduğunu göstermektedir.

## 8. Kaynakça
- Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 1330-1334. https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/
- OpenCV 4.x Documentation, Camera Calibration and 3D Reconstruction. Pinhole camera model, radial/tangential distortion coefficients and undistortion functions. https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- OpenCV 4.x Documentation, Geometric Image Transformations. Reverse mapping and cv::remap formulation. https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
- OpenCV 4.x Tutorial, Camera Calibration. Radial and tangential distortion concepts. https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Brown, D. C. (1966). Decentering distortion of lenses. Photogrammetric Engineering, 32(3), 444-462.
- Conrady, A. E. (1919). Decentred Lens-Systems. Monthly Notices of the Royal Astronomical Society, 79, 384-390.

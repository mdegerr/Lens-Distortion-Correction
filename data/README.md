# Data Directory

Bu klasör örnek çalışma sırasında ham görüntüleri yerel olarak koymak için ayrılmıştır.

Tam ham TIFF görüntüler GitHub deposuna eklenmemelidir. Bu repoda yalnızca küçük bir örnek alt küme tutulur:

```text
data/sample/
  2x/   # 2X v2 setinden 5 örnek görüntü
  3x/   # 3X v2 setinden 5 örnek görüntü
  5x/   # 5X v2 setinden 5 örnek görüntü
```

Tam veri setiyle çalışmak için önerilen yerel yapı:

```text
data/
  2x/
    2x Lens v2_*.tiff
  3x/
    3x Lens v2_*.tiff
  5x/
    5X Lens v2_*.tiff
```

Tam veri seti görüntüleri `.gitignore` tarafından dışarıda bırakılır.

# Data Directory

This directory is reserved for image data used during local experiments.

The full raw TIFF datasets are intentionally not included in this repository. Only a small public sample subset is tracked:

```text
data/sample/
  2x/   # 5 sample images from the 2X v2 dataset
  3x/   # 5 sample images from the 3X v2 dataset
  5x/   # 5 sample images from the 5X v2 dataset
```

For full local analysis, use the following recommended structure:

```text
data/
  2x/
    2x Lens v2_*.tiff
  3x/
    3x Lens v2_*.tiff
  5x/
    5X Lens v2_*.tiff
```

The complete TIFF datasets are ignored by `.gitignore` to avoid storing large raw image collections in Git.

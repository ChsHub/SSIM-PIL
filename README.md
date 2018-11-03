# SSIM-PIL
Structural similarity algorithm with compatibility to PIL.

<h2>Installation</h2>
```python
python3 -m pip install SSIM-PIL
```

<h2>Usage Example</h2>
```python
from SSIM_PIL import compare_ssim
from PIL import Image

image1 = Image.open(path)
image2 = Image.open(path)
value = compare_ssim(image1, image2)
print(value)
```
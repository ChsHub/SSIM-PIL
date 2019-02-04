# SSIM-PIL
Comparison of two images using the structural similarity algorithm (SSIM). It's compatible with the PIL.
Supports GPU acceleration via pyopencl.

## Installation
`python3 -m pip install SSIM-PIL`

## Usage Example

```python

from SSIM_PIL import compare_ssim
from PIL import Image

image1 = Image.open(path)
image2 = Image.open(path)
value = compare_ssim(image1, image2)
print(value)

```
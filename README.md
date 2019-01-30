# SSIM-PIL
Comparison of two images using the structural similarity algorithm(SSIM). It only requires with compatibility to PIL.
Version 1.0.5 supports GPU acceleration via OpenCL.
(Christoph Gohlke's distribution for windows is recommended: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl)

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
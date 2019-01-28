# SSIM-PIL
Structural similarity image comparison with compatibility to the python image library PIL.

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
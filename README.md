<a href='https://ko-fi.com/D1D7EST39' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi1.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
# SSIM-PIL
Comparison of two images using the structural similarity algorithm (SSIM).
The resulting value varies between 1.0 for identical images and 0.0 for completely different images.
It's based on the PIL and also supports GPU acceleration via pyopencl.

## Installation
`python3 -m pip install SSIM-PIL`

Be sure to install a working version of pyopencl to benefit from faster parallel execution on the GPU. (The code was
tested with OpenCl version 1.2.)

## Usage Example

```python

from SSIM_PIL import compare_ssim
from PIL import Image

image1 = Image.open(path)
image2 = Image.open(path)

# Compare images using OpenCL by default
value = compare_ssim(image1, image2)
print(value)

#  Compare images using CPU-only version
value = compare_ssim(image1, image2, GPU=False)
print(value)

```

## Feedback

For feedback please contact me over github: https://github.com/ChsHub/SSIM-PIL.

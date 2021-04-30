from os import listdir
from os.path import join

from PIL import Image
from hypothesis import given, settings
from hypothesis.strategies import integers, booleans

from SSIM_PIL import compare_ssim

image_directory = "test_images"
test_images = listdir(image_directory)


@settings(deadline=None)
@given(integers(min_value=0, max_value=len(test_images) - 1),
       integers(min_value=0, max_value=len(test_images) - 1),
       booleans(),
       integers(min_value=1, max_value=10))
def test_compare_ssim(image_0, image_1, GPU, tile_size):
    # Open images from test directory
    image_0 = Image.open(join(image_directory, test_images[image_0]))
    image_1 = Image.open(join(image_directory, test_images[image_1]))
    if image_0 == image_1:
        value = compare_ssim(image_0, image_1, tile_size, GPU)
        assert abs(value - 1.0) < 0.001


if __name__ == "__main__":
    test_compare_ssim()

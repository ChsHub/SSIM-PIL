from os import listdir
from os.path import join

from PIL import Image
from hypothesis import given, settings
from hypothesis.strategies import floats, integers
from SSIM_PIL._gpu_strategy import get_ssim_sum

image_directory = "\\test_images"
test_images = listdir(image_directory)


@settings(deadline=None)
@given(integers(min_value=0, max_value=len(test_images) - 1),
       integers(min_value=0, max_value=len(test_images) - 1),
       integers(min_value=1, max_value=min(Image.open(join(image_directory, test_images[0])).size)), floats(), floats())
def test_get_ssim_sum(image_0, image_1, tile_size, c_1, c_2):
    # Open images from test directory
    image_0 = Image.open(join(image_directory, test_images[image_0]))
    image_1 = Image.open(join(image_directory, test_images[image_1]))

    width, height = image_0.size
    pixel_len = tile_size * tile_size

    print(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2)
    assert get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2)


if __name__ == "__main__":
    test_get_ssim_sum()

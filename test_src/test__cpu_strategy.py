from os import listdir

from PIL import Image
from hypothesis import given
from hypothesis.strategies import floats, integers

from SSIM_PIL._cpu_strategy import get_ssim_sum

image_directory = '\\test_images'
test_images = listdir(image_directory)


def test__get_variance(color_count: dict, average: float):
    pass


@given(integers(min_value=0, max_value=len(test_images)), integers(min_value=0, max_value=len(test_images)),
       integers, floats, floats)
def test_get_ssim_sum(image_0, image_1, tile_size, c_1, c_2):
    image_0 = Image.open(test_images[image_0])
    image_1 = Image.open(test_images[image_1])
    width, height = image_0.size
    pixel_len = tile_size * tile_size

    assert get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2)

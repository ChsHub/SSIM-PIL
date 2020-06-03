from logging import exception
from os import listdir
from os.path import join

from PIL import Image
from hypothesis import given, settings
from hypothesis.strategies import floats, integers
from logger_default import Logger
from timerpy import Timer

from SSIM_PIL._numpy_strategy import get_ssim_sum

image_directory = "\\test_images"
test_images = listdir(image_directory)


@settings(deadline=None)
@given(integers(min_value=0, max_value=len(test_images) - 1),
       integers(min_value=0, max_value=len(test_images) - 1),
       integers(min_value=7, max_value=min(Image.open(join(image_directory, test_images[0])).size)), floats(), floats())
def test_get_ssim_sum(image_0, image_1, tile_size, c_1, c_2):
    # Open images from test directory
    print(image_0, image_1)
    with Logger():
        image_0 = Image.open(join(image_directory, test_images[image_0]))
        image_1 = Image.open(join(image_directory, test_images[image_1]))

        width, height = image_0.size
        width = (width // tile_size) * tile_size
        height = (height // tile_size) * tile_size
        pixel_len = tile_size * tile_size

        print(tile_size, pixel_len, width, height, c_1, c_2)
        try:
            with Timer():
                assert get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2)
            print()
        except Exception as e:
            exception(e)


if __name__ == "__main__":
    test_get_ssim_sum()

from collections import defaultdict
import numpy as np
from timerpy import Timer

from hypothesis import given, settings
from hypothesis.strategies import floats, integers
from SSIM_PIL._numpy_strategy import _get_image_array
from MockImage import MockImage


@given(integers(min_value=1, max_value=20), integers(min_value=1, max_value=20), integers(min_value=1, max_value=20))
def test__get_image_array(width, height, tile_size):
    width *= tile_size
    height *= tile_size

    with MockImage(size=(width, height)) as image:
        channels = len(image.image.mode)
        result = _get_image_array(image.image, width, height, tile_size=tile_size, channels=channels)
        img_height, img_width = result.shape
        assert img_height == tile_size
        assert img_width == int(height * channels * (width / tile_size))


def _get_tile_sums(image, tile_size, channels, width, height, tiles_per_row):
    pass


def _get__tile_variance(image, tile_averages, tiles_per_row, tile_size):
    pass


def get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2) -> float:
    pass


if __name__ == "__main__":
    test__get_image_array()

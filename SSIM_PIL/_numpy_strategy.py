from collections import defaultdict
import numpy as np
from timerpy import Timer


def _get_image_array(image, width, height, tile_size, channels):
    image = image.crop((0, 0, width, height))

    if 0 != image.size[0] % tile_size or 0 != image.size[1] % tile_size:
        raise ValueError

    image = np.array(image)  # Create array
    image = np.vstack(np.array_split(image, channels,
                                     axis=2))  # Separate color channels and stack into 1D (height, width, channels) -> (height * channels, width)
    image = np.reshape(image, (tile_size, image.size // tile_size))  # Reshape into 2D tile width
    return image


def _get_tile_sums(image, tile_size, channels, width, height, tiles_per_row):
    image = image.sum(axis=0)  # sum in X direction
    image = np.reshape(image, (tiles_per_row, image.size // tiles_per_row))  # Reshape 2D: tiles per row, height
    image = np.hsplit(image, (height * channels) // tile_size)
    image = np.vstack(image)
    image = image.sum(axis=1)  # sum in Y direction
    return image


def _get__tile_variance(image, tile_averages, tiles_per_row, tile_size):
    """
    TODO Calculate variances for each tile. Subtact from each row?
    :param image:
    :param tile_averages:
    :param tiles_per_row:
    :param tile_size:
    :return:
    """
    # TODO re-arange
    print(tile_averages.shape[0] * tile_size)
    # image = np.hsplit(image, image.shape[1] // tiles_per_row)
    # image = np.vstack(image)

    # image = np.hsplit(image, image.shape[1] // (tile_size * tile_size))
    # image = np.vstack(image)
    image = np.reshape(image, newshape=(1, image.shape[1], tile_size))

    # TODO Pixels - Tile wise averages

    # TODO Squared
    image = np.multiply(image, image)
    # TODO Sum tile wise


    return image


def get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2) -> float:
    """
    Get intermediate SSIM result using CPU
    :param image_0: First image
    :param image_1: Second image
    :param tile_size: Width/ height of image subsections(tiles)
    :param pixel_len: Number of pixels in the tiles
    :param width: Image width
    :param height: Image height
    :param c_1: Constant 1
    :param c_2: Constant 2
    :return: Intermediate SSIM result
    """
    # Get channels and number of tiles per row
    channels = len(image_0.mode)
    tiles_per_row = width // tile_size
    # Convert images into arrays
    image_0 = _get_image_array(image_0, width, height, tile_size, channels)
    image_1 = _get_image_array(image_1, width, height, tile_size, channels)
    covariances = np.multiply(image_0, image_1, dtype=np.uint16)
    # Tile sums
    sums_0 = _get_tile_sums(image_0, tile_size, channels, width, height, tiles_per_row)
    sums_1 = _get_tile_sums(image_1, tile_size, channels, width, height, tiles_per_row)
    covariances = _get_tile_sums(covariances, tile_size, channels, width, height, tiles_per_row)
    # Get covariances
    covariances = np.subtract(covariances, np.divide(np.multiply(sums_0, sums_1), pixel_len))
    covariances = np.divide(covariances, pixel_len)
    # Get averages
    average_0 = np.divide(sums_0, pixel_len)
    average_1 = np.divide(sums_1, pixel_len)
    # Todo Get variances
    variance_0_1 = _get__tile_variance(image_0, average_0, tiles_per_row, tile_size)
    variance_0_1 += _get__tile_variance(image_1, average_1, tiles_per_row, tile_size)

    # TODO Get final result

    ssim_sum = 1.0
    return ssim_sum

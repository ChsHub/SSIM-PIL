from collections import defaultdict


def _get_variance(color_count: dict, average: float) -> float:
    """
    Compute the variance
    :param color_count: Number of each color in the tile
    :param average: Average pixel color in the tile
    :return: Variance
    """
    variance = 0.0
    for pixel in color_count:
        a = pixel - average
        variance += color_count[pixel] * a * a
    return variance


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
    channels = range(len(image_0.mode))
    ssim_sum = 0.0

    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):

            # Get pixel tile
            box = (x, y, x + tile_size, y + tile_size)
            tile_0 = image_0.crop(box)
            tile_1 = image_1.crop(box)

            for i in channels:
                # Tile
                pixel0, pixel1 = tile_0.getdata(band=i), tile_1.getdata(band=i)
                color_count_0 = defaultdict(int)
                color_count_1 = defaultdict(int)
                covariance = 0.0

                for i1, i2 in zip(pixel0, pixel1):
                    color_count_0[i1] += 1
                    color_count_1[i2] += 1
                    covariance += i1 * i2

                pixel_sum_0 = sum(pixel0)
                pixel_sum_1 = sum(pixel1)
                average_0 = pixel_sum_0 / pixel_len
                average_1 = pixel_sum_1 / pixel_len

                covariance = (covariance - pixel_sum_0 * pixel_sum_1 / pixel_len) / pixel_len
                # Calculate the sum of both image's variances
                variance_0_1  = _get_variance(color_count_0, average_0)
                variance_0_1 += _get_variance(color_count_1, average_1)
                variance_0_1 /= pixel_len

                ssim_sum += (2.0 * average_0 * average_1 + c_1) * (2.0 * covariance + c_2) / (
                        average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0_1 + c_2)

    return ssim_sum

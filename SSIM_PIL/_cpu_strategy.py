from collections import defaultdict

def _get_variance(color_count: dict, average: float, pixel_len: int) -> float:
    """
    Compute the variance
    :param color_count: Number of each color in the tile
    :param average: Average pixel color in the tile
    :param pixel_len: Number of pixels in the tile
    :return: Variance
    """
    variance = 0
    for pixel in color_count:
        a = pixel - average
        variance += color_count[pixel] * a * a
    return variance / pixel_len


def get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2) -> float:

    channels = range(len(image_0.mode))
    ssim_sum = 0

    for x, xw in zip(range(0, width, tile_size), range(tile_size, width + tile_size, tile_size)):
        for y, yw in zip(range(0, height, tile_size), range(tile_size, height + tile_size, tile_size)):
            box = (x, y, xw, yw)
            tile_0 = image_0.crop(box)
            tile_1 = image_1.crop(box)

            for i in channels:
                # tile
                pixel0, pixel1 = tile_0.getdata(band=i), tile_1.getdata(band=i)
                color_count_0 = defaultdict(int)
                color_count_1 = defaultdict(int)
                covariance = 0

                for i1, i2 in zip(pixel0, pixel1):
                    color_count_0[i1] += 1
                    color_count_1[i2] += 1
                    covariance += i1 * i2

                pixel_sum_0 = sum(pixel0)
                pixel_sum_1 = sum(pixel1)
                average_0 = pixel_sum_0 / pixel_len
                average_1 = pixel_sum_1 / pixel_len

                covariance = (covariance - pixel_sum_0 * pixel_sum_1 / pixel_len) / pixel_len
                variance_0 = _get_variance(color_count_0, average_0, pixel_len)
                variance_1 = _get_variance(color_count_1, average_1, pixel_len)

                ssim_sum += (2 * average_0 * average_1 + c_1) * (2 * covariance + c_2) / (
                        average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0 + variance_1 + c_2)

    return ssim_sum
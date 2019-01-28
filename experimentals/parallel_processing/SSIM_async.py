from asyncio import get_event_loop, ensure_future
from collections import defaultdict

from utility.timer import Timer


def variance(color_count, average, pixel_len):
    # https://en.wikipedia.org/wiki/Standard_deviation#Population_standard_deviation_of_grades_of_eight_students
    variance = 0
    for pixel, count in zip(color_count.keys(), color_count.values()):
        a = pixel - average
        variance += count * a * a
    return variance / pixel_len


def _ssim_tile(pixel0, pixel1, pixel_len, c_1, c_2):
    # https: // en.wikipedia.org / wiki / Structural_similarity  # Algorithm

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
    variance_0 = variance(color_count_0, average_0, pixel_len)
    variance_1 = variance(color_count_1, average_1, pixel_len)

    return (2 * average_0 * average_1 + c_1) * (2 * covariance + c_2) / (
            average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0 + variance_1 + c_2)


async def run_band(x, height, window_size, image_0, image_1, pixel_len, c_1, c_2):
    ssim = 0
    for y in range(0, height - height % window_size, window_size):
        tile_0 = image_0.crop((x, y, x + window_size, y + window_size))
        tile_1 = image_1.crop((x, y, x + window_size, y + window_size))
        for i in range(len(tile_0.mode)):
            ssim += _ssim_tile(tile_0.getdata(band=i), tile_1.getdata(band=i), pixel_len, c_1, c_2)
    return ssim


async def run(image_0, image_1):
    with Timer('PREP'):
        window_size = 7
        dynamic_range = 255  # *3
        c_1 = (dynamic_range * 0.01) ** 2
        c_2 = (dynamic_range * 0.03) ** 2
        pixel_len = window_size * window_size
        width, height = image_0.size
        ssim = []

    for x in range(0, width - width % window_size, window_size):
        ssim.append(ensure_future(
            run_band(x, height, window_size, image_0, image_1, pixel_len, c_1, c_2)
        ))

    return sum([await future for future in ssim]) / len(image_0.mode) / (width // window_size * height // window_size)


def SSIM(image_0, image_1):
    if image_0.size != image_1.size:
        print('ERROR images are not same size')
        return

    with Timer('ASYNC'):
        loop = get_event_loop()
        return loop.run_until_complete(run(image_0, image_1))

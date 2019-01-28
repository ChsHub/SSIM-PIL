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

    pixel_sum_0 = sum(pixel0)
    pixel_sum_1 = sum(pixel1)
    # pixel_sum_0 = 0
    # pixel_sum_1 = 0

    covariance = 0
    for i1, i2 in zip(pixel0, pixel1):
        # i1 = sum(i1)
        # i2 = sum(i2)
        # pixel_sum_0 += i1
        # pixel_sum_1 += i2

        color_count_0[i1] += 1
        color_count_1[i2] += 1
        covariance += i1 * i2

    average_0 = pixel_sum_0 / pixel_len
    average_1 = pixel_sum_1 / pixel_len

    covariance = (covariance - pixel_sum_0 * pixel_sum_1 / pixel_len) / pixel_len
    variance_0 = variance(color_count_0, average_0, pixel_len)
    variance_1 = variance(color_count_1, average_1, pixel_len)

    return (2 * average_0 * average_1 + c_1) * (2 * covariance + c_2) / (average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0 + variance_1 + c_2)


class Runner:
    def __init__(self, width, window_size, image_0, image_1, pixel_len, c_1, c_2):
        self.width = width
        self.window_size = window_size
        self.image_0 = image_0
        self.image_1 = image_1
        self.pixel_len = pixel_len
        self.c_1 = c_1
        self.c_2 = c_2

    def ssim(self, xy):
        s = 0
        for x in range(0, self.width - self.width % self.window_size, self.window_size):
            s += _ssim_tile(*get_pixels(x, xy[1], self.width, self.window_size, self.image_0, self.image_1),
                            self.pixel_len, self.c_1, self.c_2)
        return s


def get_pixels(x, y, width, window_size, image_0, image_1):
    pixel0 = []
    pixel1 = []
    for w in range(y * width, (y + window_size) * width, width):
        pixel0 += image_0[x + w:x + window_size + w]
        pixel1 += image_1[x + w:x + window_size + w]
    return pixel0, pixel1


def SSIM(image_0, image_1):
    if image_0.size != image_1.size:
        print('ERROR images are not same size')
        return
    # no else
    with Timer('PREP'):
        window_size = 7
        dynamic_range = 255  # *3
        c_1 = (dynamic_range * 0.01) ** 2
        c_2 = (dynamic_range * 0.03) ** 2
        pixel_len = window_size * window_size
        width, height = image_0.size
        # image_1 = image_0
        ssim = 0
        max_workers = 2
        divider = 1

    with Timer('SINGLE'):
        for i, s in enumerate(image_0.mode):

            band_0 = list(image_0.getdata(band=i))
            band_1 = list(image_1.getdata(band=i))
            sss = 0
            for x in range(0, width - width % window_size, window_size):
                for y in range(0, height - height % window_size, window_size):
                    sss += _ssim_tile(*get_pixels(x, y, width, window_size, band_0, band_1), pixel_len, c_1, c_2)
            sss /= (width // window_size * height // window_size)
            ssim += sss
        ssim = ssim / len(image_0.mode)
    return ssim
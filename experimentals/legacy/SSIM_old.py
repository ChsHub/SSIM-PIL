from collections import defaultdict


def _get_pixel_array(window_size, position, pixels):
    result = []
    for y in range(position[1], position[1] + window_size):
        for x in range(position[0], position[0] + window_size):
            result.append(pixels[x, y])

    return result


def _get_pixel_probabilities(pixels):
    color_count = defaultdict(int)
    for pixel in pixels:
        color_count[pixel] += 1

    color_sum = sum(pixels)
    return [color_count[pixel] / color_sum for pixel in pixels]


# https://en.wikipedia.org/wiki/Expected_value
def get_expected_value(pixels):
    """
    :param pixels:
    :return: expected value, probabilities of each pixel value
    """

    pixel_probabilities = _get_pixel_probabilities(pixels)
    # SIGMA x * p
    return sum([v * p for v, p in zip(pixels, pixel_probabilities)]), pixel_probabilities


def variance(pixels):
    expected_value, pixel_probabilities = get_expected_value(pixels)

    return sum([p * (v - expected_value) ** 2 for v, p in zip(pixels, pixel_probabilities)]), expected_value


def covariance(pixels, E_x, E_y):
    expected_value = get_expected_value(pixels)[0]
    return expected_value - E_x * E_y


def average(pixels, pixel_len):
    return sum(pixels) / pixel_len


def _ssim_tile(img_x, img_y, dynamic_range, pixel_len, window_size, position):
    # https: // en.wikipedia.org / wiki / Structural_similarity  # Algorithm

    pix_x = _get_pixel_array(window_size, position, img_x)
    pix_y = _get_pixel_array(window_size, position, img_y)

    avg_x = average(pix_x, pixel_len)
    avg_y = average(pix_y, pixel_len)
    var_x, E_x = variance(pix_x)
    var_y, E_y = variance(pix_y)
    cov = covariance(pix_x + pix_y, E_x, E_y)

    c_1 = (dynamic_range * 0.01) ** 2
    c_2 = (dynamic_range * 0.03) ** 2

    return (2 * avg_x * avg_y + c_1) * (2 * cov + c_2) / (avg_x ** 2 + avg_y ** 2 + c_1) / (var_x + var_y + c_2)


def SSIM(image_0, image_1):
    if image_0.size != image_1.size:
        print('ERROR images are not same size')
        return
    # no else

    window_size = 8
    ssim, i = 0, 0
    dynamic_range = 255  # *3
    pixel_len = window_size * window_size
    width, height = image_0.size
    image_0 = image_0.split()[0].load()
    image_1 = image_1.split()[0].load()

    for x in range(0, width - width % window_size, window_size):
        for y in range(0, height - height % window_size, window_size):
            ssim += _ssim_tile(image_0, image_1, dynamic_range, pixel_len, window_size=window_size, position=(x, y))
            i += 1
    print(i)
    return ssim / i

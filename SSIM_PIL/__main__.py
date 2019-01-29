from collections import defaultdict


# https://en.wikipedia.org/wiki/Standard_deviation#Population_standard_deviation_of_grades_of_eight_students
def variance(color_count, average, pixel_len):
    variance = 0
    for pixel in color_count:
        a = pixel - average
        variance += color_count[pixel] * a * a
    return variance / (pixel_len - 1)


# https: // en.wikipedia.org / wiki / Structural_similarity  # Algorithm
def compare_ssim(image_0, image_1, tile_size=7, GPU=False) -> float:
    """
    Compute the structural similarity between the two images.
    :param image_0: PIL Image object
    :param image_1: PIL Image object
    :param tile_size: Height and width of the image's sub-sections used
    :param GPU: If true, try to compute on GPU
    :return: Structural similarity value
    """

    if image_0.size != image_1.size:
        raise AttributeError('The images do not have the same resolution')
    # no else
    if image_0.mode != image_1.mode:
        raise AttributeError('The images have different color channels')
    # no else

    pixel_len = tile_size * tile_size
    dynamic_range = 255
    c_1 = (dynamic_range * 0.01) ** 2
    c_2 = (dynamic_range * 0.03) ** 2
    width, height = image_0.size

    if width < tile_size or height < tile_size:
        raise AttributeError('The images are smaller than the window_size')
    # no else

    if GPU:
        import gpu_ssim
        # return

        ssim = gpu_ssim.compare(image_0, image_1, tile_size, width, height, c_1, c_2)
    # no else

    width = width // tile_size * tile_size
    height = height // tile_size * tile_size
    channels = range(len(image_0.mode))
    ssim = 0


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
                variance_0 = variance(color_count_0, average_0, pixel_len)
                variance_1 = variance(color_count_1, average_1, pixel_len)

                ssim += (2 * average_0 * average_1 + c_1) * (2 * covariance + c_2) / (
                        average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0 + variance_1 + c_2)

    return ssim / (len(image_0.mode) * width * height) * tile_size * tile_size


if __name__ == "__main__":
    from PIL import Image as pil_Image
    compare_ssim(pil_Image.open("D:\Bilder/volume.jpg"), pil_Image.open("D:\Bilder/volume.jpg"), GPU=True)
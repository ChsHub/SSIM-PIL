from pathlib import Path

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags

"""
https://stackoverflow.com/a/26395800/7062162
https://github.com/smistad/OpenCL-Gaussian-Blur/blob/master/gaussian_blur.cl
https://gist.github.com/likr/3735779
http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
https://www.johndcook.com/blog/2008/09/26/comparing-three-methods-of-computing-standard-deviation/
https://benshope.blogspot.com/2013/11/pyopencl-tutorial-part-1-introspection.html

https://stackoverflow.com/questions/20613013/opencl-float-sum-reduction

https://us.fixstars.com/opencl/book/OpenCLProgrammingBook/calling-the-kernel/
https://cnugteren.github.io/tutorial/pages/page14.html
http://developer.download.nvidia.com/compute/DevZone/docs/html/OpenCL/doc/OpenCL_Programming_Guide.pdf
https://cims.nyu.edu/~schlacht/OpenCLModel.pdf
"""


# TODO test maximum image sizes

def _create_context():
    """
    Search for a device with OpenCl support, and create device context
    :return: First found device or raise EnvironmentError if none is found
    """
    platforms = cl.get_platforms()  # Select the first platform [0]
    if not platforms:
        raise EnvironmentError('No openCL platform (or driver) available.')

    # Return first found device
    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            return cl.Context([devices[0]])

    raise EnvironmentError('No openCL devices (or driver) available.')


def _create_program(context):
    """
    Load the OpenCL source code and compile the program
    :param context: Device context
    :return: Compiled program
    """

    source = Path('CL_SOURCE.c').read_text()
    source, CL_SOURCE_per_pixel, end = source.split('/* SPLIT CODE HERE */')

    # Copy paste the openCL code for all color dimensions since there is no pixel access
    source = [source]
    for i, dim in enumerate(['.x', '.y', '.z', '.w']):
        source.append(CL_SOURCE_per_pixel.replace('.x', dim).replace('get_global_id(0)==0',
                                                                     'get_global_id(0)==' + str(i)))
    source = ''.join(source) + end
    # Compile OpenCL program
    return cl.Program(context, source).build()


# Saving context, queue and the compiled program saves time during repeated function calls
_context = _create_context()
_queue = cl.CommandQueue(_context)
_program = _create_program(_context)


def _get_image_buffer(image):
    """
    Create the buffer object for a image
    :param image: PIL image object
    :return: CL buffer object
    """
    image = image.convert("RGBA")
    image = np.array(image)
    return cl.image_from_array(_context, image, num_channels=4, mode="r", norm_int=False)
    # pyopencl._cl.LogicError: clCreateImage failed: INVALID_IMAGE_SIZE
    # TODO large images fail


def get_ssim_sum(image_0, image_1, tile_size: int, pixel_len: int, width: int, height: int, c_1: float,
                 c_2: float) -> float:
    """
    Get intermediate SSIM result using GPU
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

    # Used variables
    color_channels = len(image_0.mode)
    width //= tile_size
    height //= tile_size

    # Convert images to numpy array and create buffer object
    image_0 = _get_image_buffer(image_0)
    image_1 = _get_image_buffer(image_1)

    # Initialize result vector, with length equal to the number of tiles.
    result = np.zeros(shape=width * height * color_channels, dtype=np.float32)
    result_buffer = cl.Buffer(_context, mem_flags.WRITE_ONLY, result.nbytes)

    # Run the program with one thread for every tile in every color
    _program.convert(_queue, (color_channels, width, height), None,
                     image_0, image_1, result_buffer, np.int32(tile_size),
                     np.int32(width), np.int32(height), np.float32(pixel_len),
                     np.float32(c_1), np.float32(c_2))

    # Copy result
    cl.enqueue_copy(_queue, result, result_buffer)
    return result.sum()

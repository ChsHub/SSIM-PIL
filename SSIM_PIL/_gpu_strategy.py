from __future__ import absolute_import, print_function

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags
from utility.timer import Timer # TODO REMOVE TIMING

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
            max_work_item_sizes = devices[0].max_work_item_sizes
            return cl.Context([devices[0]]), max_work_item_sizes

    raise EnvironmentError('No openCL devices (or driver) available.')


# Saving context saves time during repeated function calls
_context, _max_work_item_sizes = _create_context()


with open(__file__.replace('_gpu_strategy.py', 'cl_code.cpp'), mode='r') as f:
    CL_SOURCE = f.read()


def create_image_array(image, height, width):
    image_0_array = np.empty(shape=(height, width, 4), dtype=np.uint8)
    image_0_array[0:height, 0:width, 0:len(image.mode)] = np.array(image)[0:height, 0:width, 0:len(image.mode)]
    return image_0_array


def get_ssim_sum(image_0, image_1, tile_size, pixel_len, width, height, c_1, c_2) -> (float, bool):
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

    color_channels = len(image_0.mode)
    # Number of threads per work group has to be dividable by number of pixels per tile
    work_group_size = _max_work_item_sizes[0] - (_max_work_item_sizes[0] % pixel_len)
    thread_num = (width * height) # One thread for each pixel
    # If number of threads can't be divided, add additional threads
    if thread_num % work_group_size:
        thread_num = thread_num - (thread_num % work_group_size) + work_group_size
    # no else

    # Copy paste the openCL code for all color dimensions since there is no pixel access
    source = CL_SOURCE  # .split('/*XXXXXXX*/')
    # for i, dim in enumerate(['.y', '.z', '.w'][:len(image_0.mode) - 1]):
    #    source.insert(2, source[1].replace('.x', dim).replace('get_global_id(0)==0', 'get_global_id(0)=='+str(i+1)))
    # source = ''.join(source)
    source = source.replace('[pixel_len]', '[42]')
    source = source.replace('BLOCK_SIZE', str(work_group_size))

    with open('cl_code compiled.cpp', mode='w') as f:
        f.write(source)

    # Create a context with selected device
    context = _context
    queue = cl.CommandQueue(context)

    # Convert images to numpy array and create buffer object
    # TODO: how to buffer RGB images
    with Timer('CONVERT IMAGES'):  # TODO REMOVE TIMING
        image_0 = image_0.convert("RGBA")
        image_0 = cl.image_from_array(context, np.array(image_0), len(image_0.mode), "r", norm_int=False)

        image_1 = image_1.convert("RGBA")
        image_1 = cl.image_from_array(context, np.array(image_1), len(image_1.mode), "r", norm_int=False)

    # Initialize result vector with zeroes, because tile results are added.
    result = np.zeros(shape=(width // tile_size) * (height // tile_size), dtype=np.float32)
    result_buffer = cl.Buffer(context, mem_flags.WRITE_ONLY, result.nbytes)

    # Compile and run openCL program
    with Timer('BUILD CL PROGRAM'):  # TODO REMOVE TIMING
        program = cl.Program(context, source).build()

    with Timer('RUN CL PROGRAM'):  # TODO REMOVE TIMING
        program.convert(queue, (thread_num, 1), (work_group_size, 1),
                        image_0, image_1,
                        result_buffer,
                        np.int32(tile_size),
                        np.int32(width), np.int32(height), np.float32(pixel_len),
                        np.float32(c_1), np.float32(c_2))
    with Timer('Copy RESULT'): # TODO REMOVE TIMING
        # Copy result
        cl.enqueue_copy(queue, result, result_buffer)

    with Timer('SUM RESULT'): # TODO REMOVE TIMING
            ssim_sum = result.sum()
    return ssim_sum

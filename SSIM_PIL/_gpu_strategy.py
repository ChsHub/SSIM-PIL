from __future__ import absolute_import, print_function

from copy import copy

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags
from utility.timer import Timer


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


# Saving context saves time during repeated function calls
_context = _create_context()

"""
https://stackoverflow.com/a/26395800/7062162
https://github.com/smistad/OpenCL-Gaussian-Blur/blob/master/gaussian_blur.cl
https://gist.github.com/likr/3735779
http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
https://www.johndcook.com/blog/2008/09/26/comparing-three-methods-of-computing-standard-deviation/
https://benshope.blogspot.com/2013/11/pyopencl-tutorial-part-1-introspection.html
"""

CL_SOURCE = '''//CL//
__kernel void convert(
    __read_only image2d_t img0, 
    __read_only image2d_t img1, 
    __global float *res_g,
    __const int tile_size,
    __const int width,
    __const float pixel_len,
    const float c_1,
    const float c_2)
{
    // Create sampler and image position for pixel access
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(1), get_global_id(2)) * tile_size;

    /*XXXXXXX*/
    if(get_global_id(0)==0)
    {
        uint4 pix0;
        uint4 pix1;
        int pixel_sum_0 = 0;
        int pixel_sum_1 = 0;
        float covariance = 0;
        float variance_0 = 0;
        float variance_1 = 0;
        __local uint4 pixels0[pixel_len][pixel_len];
        __local uint4 pixels1[pixel_len][pixel_len];
        
        for(int x=pos. x; x < (pos. x + tile_size); x++) // Space between . and x prevents replacing during copy
        {
            for(int y=pos. y; y < (pos. y + tile_size); y++)
            {
    
                pix0 = read_imageui(img0, sampler, (int2)(x, y));
                pix1 = read_imageui(img1, sampler, (int2)(x, y));
                pixels0[y * tile_size + x][0] = pix0;
                pixels1[y * tile_size + x][0] = pix1;
                pixel_sum_0 += pix0.x;
                pixel_sum_1 += pix1.x;
                covariance += pix0.x * pix1.x;
            }
        }
        
        // Calculate covariance
        covariance = (covariance - pixel_sum_0 * pixel_sum_1 / pixel_len) / pixel_len;
        float average_0 = (float)pixel_sum_0 / pixel_len;
        float average_1 = (float)pixel_sum_1 / pixel_len;
        float temp;
        
        for(int x=pos. x; x < (pos. x + tile_size); x++)
        {
            for(int y=pos. y; y < (pos. y + tile_size); y++)
            {
                pix0 = read_imageui(img0, sampler, (int2)(x, y));
                pix1 = read_imageui(img1, sampler, (int2)(x, y));
                
                temp = (float)pix0.x - average_0;
                variance_0 += temp * temp;
                
                temp = (float)pix1.x - average_1;
                variance_1 += temp * temp;
            }
        }
        variance_0 /= pixel_len;
        variance_1 /= pixel_len;
        
        res_g[get_global_id(1) + get_global_id(2) * width] += (2.0 * average_0 * average_1 + c_1) * (2.0 * covariance + c_2) 
                                / (average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0 + variance_1 + c_2);
    }
    /*XXXXXXX*/
}
'''


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

    with Timer('PREPARE SOURCE'):
        # Copy paste the openCL code for all color dimensions since there is no pixel access

        source = CL_SOURCE.split('/*XXXXXXX*/')
        for i, dim in enumerate(['.y', '.z', '.w'][:len(image_0.mode) - 1]):
            source.insert(2, source[1].replace('.x', dim).replace('get_global_id(0)==0', 'get_global_id(0)=='+str(i+1)))
        source = ''.join(source)
        source = source.replace('[pixel_len]', '[42]')
        with open('cl_code.cpp', mode='w') as f:
            f.write(source)
    with Timer('CREATE CONTEXT'):
        # Create a context with selected device
        context = _context
        queue = cl.CommandQueue(context)

    # Convert images to numpy array and create buffer object
    # TODO: how to buffer RGB images
    with Timer('CONVERT IMAGES'):
        image_0 = image_0.convert("RGBA")
        image_0 = cl.image_from_array(context, np.array(image_0), len(image_0.mode), "r", norm_int=False)

        image_1 = image_1.convert("RGBA")
        image_1 = cl.image_from_array(context, np.array(image_1), len(image_1.mode), "r", norm_int=False)

    width //= tile_size
    height //= tile_size
    # Initialize result vector with zeroes, because tile results are added.
    result = np.zeros(shape=width * height, dtype=np.float32)
    result_buffer = cl.Buffer(context, mem_flags.WRITE_ONLY, result.nbytes)

    # Compile and run openCL program
    with Timer('RUN CL PROGRAM'):
        program = cl.Program(context, source).build()
        program.convert(queue, (color_channels, width, height), None,
                        image_0, image_1, result_buffer, np.int32(tile_size),
                        np.int32(width), np.float32(pixel_len),
                        np.float32(c_1), np.float32(c_2))
    with Timer('Copy RESULT'):
        # Copy result
        cl.enqueue_copy(queue, result, result_buffer)

    with Timer('SUM RESULT'):
        ssim_sum = result.sum()
    return ssim_sum

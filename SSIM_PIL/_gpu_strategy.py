from __future__ import absolute_import, print_function

from time import time

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags
from utility.timer import Timer # TODO remove

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
            return cl.Context([devices[0]])

    raise EnvironmentError('No openCL devices (or driver) available.')


# Saving context saves time during repeated function calls
_context = _create_context()
# Create a context with selected device
queue = cl.CommandQueue(_context)

CL_SOURCE = '''//CL//
__kernel void convert(
    __read_only image2d_t img0, 
    __read_only image2d_t img1, 
    __global float *res_g,
    __const int tile_size,
    __const int width, __const int height,
    __const float pixel_len,
    const float c_1,
    const float c_2)
{

    '''
CL_SOURCE_per_pixel = '''
    if(get_global_id(0)==0)
    {
        // Create sampler and image position for pixel access
        const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        int pos_x = get_global_id(1) * tile_size;
        int pos_y = get_global_id(2) * tile_size;


        ushort pix[2];
        int pixel_sum[2];
        pixel_sum[0] = 0;
        pixel_sum[1] = 0;
        float covariance = 0.0;

        
        for(int x = pos_x; x < pos_x + tile_size; x++)
        {
            for(int y = pos_y; y < pos_y + tile_size; y++)
            {
                
                pix[0] = read_imageui(img0, sampler, (int2)(x, y)).x;
                pixel_sum[0] += pix[0];
                
                pix[1] = read_imageui(img1, sampler, (int2)(x, y)).x;
                pixel_sum[1] += pix[1];
                
                covariance += pix[0] * pix[1];
            }
        }
        
        // Calculate covariance
        covariance = (covariance - pixel_sum[0] * pixel_sum[1] / pixel_len) / pixel_len;
        
        float average[2];
        average[0] = (float)pixel_sum[0] / pixel_len;
        average[1] = (float)pixel_sum[1] / pixel_len;
        
        // Calculate sum the two images variances
        float variance_0_1_sum = 0.0;
        float temp_pix;
        
        for(int x = pos_x; x < pos_x + tile_size; x++)
        {
            for(int y = pos_y; y < pos_y + tile_size; y++)
            {
                temp_pix = read_imageui(img0, sampler, (int2)(x, y)).x;
                temp_pix = temp_pix - average[0];
                variance_0_1_sum += temp_pix * temp_pix;
                
                temp_pix = read_imageui(img1, sampler, (int2)(x, y)).x;
                temp_pix = temp_pix - average[1];
                variance_0_1_sum += temp_pix * temp_pix;
            }
        }
        
        // Calculate the final SSIM value
        
        res_g[get_global_id(0) * width * height + (get_global_id(1) + get_global_id(2) * width)] = 
        (2.0 * average[0] * average[1] + c_1) * (2.0 * covariance + c_2)
        / (average[0] * average[0] + average[1] * average[1] + c_1) / (variance_0_1_sum / pixel_len + c_2);
    }
'''
# Copy paste the openCL code for all color dimensions since there is no pixel access
source = [CL_SOURCE]
for i, dim in enumerate(['.x', '.y', '.z', '.w']):
    source.append(
        CL_SOURCE_per_pixel.replace('.x', dim).replace('get_global_id(0)==0', 'get_global_id(0)==' + str(i)))
source = ''.join(source) + '}'
# Compile OpenCL program
program = cl.Program(_context, source).build()


def _get_image_buffer(context, image):

    image = image.convert("RGBA")
    image = np.array(image)
    return cl.image_from_array(context, image, num_channels=4, mode="r", norm_int=False)

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

    # Convert images to numpy array and create buffer object
    image_0 = _get_image_buffer(_context, image_0)
    image_1 = _get_image_buffer(_context, image_1)

    width //= tile_size
    height //= tile_size
    # Initialize result vector with zeroes, because tile results are added.

    result = np.zeros(shape=width * height * color_channels, dtype=np.float32)
    result_buffer = cl.Buffer(_context, mem_flags.WRITE_ONLY, result.nbytes)

    program.convert(queue, (color_channels, width, height), None,
                    image_0, image_1, result_buffer, np.int32(tile_size),
                    np.int32(width), np.int32(height), np.float32(pixel_len),
                    np.float32(c_1), np.float32(c_2))

    # Copy result
    cl.enqueue_copy(queue, result, result_buffer)
    return result.sum()

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

"""
https://stackoverflow.com/a/26395800/7062162
https://github.com/smistad/OpenCL-Gaussian-Blur/blob/master/gaussian_blur.cl
https://gist.github.com/likr/3735779
http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
https://www.johndcook.com/blog/2008/09/26/comparing-three-methods-of-computing-standard-deviation/
https://benshope.blogspot.com/2013/11/pyopencl-tutorial-part-1-introspection.html
"""

CL_SOURCE = '''//CL//
__kernel void convert(read_only image2d_t img0, read_only image2d_t img1, __global float *res_g,
    const int tile_size,
    const int width,
    const float pixel_len,
    const float c_1,
    const float c_2)
{
    // Create sampler and image position for pixel access
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1)) * tile_size;

    /*XXXXXXX*/
    {
    uint4 pix0;
    uint4 pix1;
    int pixel_sum_0 = 0;
    int pixel_sum_1 = 0;
    float covariance = 0;
    float variance_0 = 0;
    float variance_1 = 0;
    
    for(int x=pos. x; x < (pos. x + tile_size); x++) // Space between . and x prevents replacing during copy
    {
        for(int y=pos. y; y < (pos. y + tile_size); y++)
        {
            pix0 = read_imageui(img0, sampler, (int2)(x, y));
            pix1 = read_imageui(img1, sampler, (int2)(x, y));
            pixel_sum_0 += pix0.x;
            pixel_sum_1 += pix1.x;
            covariance += pix0.x * pix1.x;
        }
    }
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
    
    res_g[get_global_id(0) + get_global_id(1) * width] += (2.0 * average_0 * average_1 + c_1) * (2.0 * covariance + c_2) 
                            / (average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0 + variance_1 + c_2);
    }
    /*XXXXXXX*/
}
'''
# Copy paste the openCL code for all dimensions since there is no pixel access
CL_SOURCE = CL_SOURCE.split('/*XXXXXXX*/')
CL_SOURCE.insert(1, CL_SOURCE[1].replace('.x','.y'))
CL_SOURCE.insert(1, CL_SOURCE[1].replace('.x','.z'))
CL_SOURCE = ''.join(CL_SOURCE)

def print_platform_info():
    print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
    for platform in cl.get_platforms():  # Print each platform on this computer
        print('=' * 60)
        print('Platform - Name:  ' + platform.name)
        print('Platform - Vendor:  ' + platform.vendor)
        print('Platform - Version:  ' + platform.version)
        print('Platform - Profile:  ' + platform.profile)
        for device in platform.get_devices():  # Print each device per-platform
            print('    ' + '-' * 56)
            print('    Device - Name:  ' + device.name)
            print('    Device - Type:  ' + cl.device_type.to_string(device.type))
            print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
            print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
            print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size / 1024))
            print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size / 1024))
            print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size / 1073741824.0))
    print('\n')

def compare(image_0, image_1, tile_size, width, height, c_1, c_2) -> (float, bool):

    platform = cl.get_platforms()[0]  # Select the first platform [0]
    device = platform.get_devices()[0]  # Select the first device on this platform [0]
    context = cl.Context([device])  # Create a context with your device
    queue = cl.CommandQueue(context)
    width //= tile_size
    height //= tile_size

    # convert images to numpy array and create buffer object
    # TODO: buffer RGB images
    image_0 = cl.image_from_array(context, np.array(image_0.convert("RGBA")), 4, "r", norm_int=False)
    image_1 = cl.image_from_array(context, np.array(image_1.convert("RGBA")), 4, "r", norm_int=False)
    result = np.zeros(shape=width * height, dtype=np.float32) # TODO change height // tile
    result_buffer = cl.Buffer(context, mf.WRITE_ONLY, result.nbytes)

    # Compile and run openCL program
    program = cl.Program(context, CL_SOURCE).build()
    program.convert(queue, (width, height), None,
                    image_0, image_1, result_buffer, np.int32(tile_size),
                    np.int32(width), np.float32(tile_size * tile_size),
                    np.float32(c_1), np.float32(c_2))
    # Receive result
    cl.enqueue_copy(queue, result, result_buffer)
    ssim = result.sum()
    # print(result)

    # TODO Error message and test for devices
    return ssim, False

print_platform_info()
from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

"""
https://github.com/smistad/OpenCL-Gaussian-Blur/blob/master/gaussian_blur.cl
https://gist.github.com/likr/3735779
http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
https://www.johndcook.com/blog/2008/09/26/comparing-three-methods-of-computing-standard-deviation/
"""
CL_SOURCE = '''//CL//
__kernel void convert(
    read_only image2d_t img0, read_only image2d_t img1,
    __global float *res_g,
    const int tile_size,
    const int width,
    const int pixel_len,
    const float c_1,
    const float c_2)
{
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1)) * tile_size;
    uint4 pix0;
    uint4 pix1;
    float covariance = 0;
    int pixel_sum_0 = 0;
    int pixel_sum_1 = 0;
        
    for(int x=0; x < tile_size; x++)
    {
        for(int y=0; y < tile_size; y++)
        {
            pix0 = read_imageui(img0, sampler, pos  + (int2)(x, y));
            pix1 = read_imageui(img1, sampler, pos  + (int2)(x, y));
            pixel_sum_0 += pix0.x;
            pixel_sum_1 += pix1.x;
            covariance += pix0.x * pix1.x;
        }
    }
    covariance = (covariance - pixel_sum_0 * pixel_sum_1 / pixel_len) / pixel_len;
    float average_0 = pixel_sum_0 / pixel_len;
    float average_1 = pixel_sum_1 / pixel_len;
    
    float variance_0 = 0;
    float variance_1 = 0;
    for(int x=0; x < tile_size; x++)
    {
        for(int y=0; y < tile_size; y++)
        {
            pix0 = read_imageui(img0, sampler, pos  + (int2)(x, y));
            pix1 = read_imageui(img1, sampler, pos  + (int2)(x, y));
            int a = pix0.x - average_0;
            variance_0 += a * a;
            a = pix1.x - average_1;
            variance_1 += a * a;
        }
    }
    
    variance_0 /= (pixel_len - 1);
    variance_1 /= (pixel_len - 1);
    
    res_g[get_global_id(0) * width + get_global_id(1)] = average_0;
     
    res_g[get_global_id(0) * width + get_global_id(1)] = (2 * average_0 * average_1 + c_1) * (2 * covariance + c_2) 
                            / (average_0 * average_0 + average_1 * average_1 + c_1) / (variance_0 + variance_1 + c_2);
}
'''


def compare(image_0, image_1, tile_size, width, height, c_1, c_2) -> int:


    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    image_0 = image_0.convert("RGBA")
    img0 = cl.image_from_array(context, np.array(image_0), len(image_0.mode), "r", norm_int=False)

    image_1 = image_1.convert("RGBA")
    img1 = cl.image_from_array(context, np.array(image_1), len(image_1.mode), "r", norm_int=False)

    result_np = np.empty(shape=width * height, dtype=np.float32)
    res_g = cl.Buffer(context, mf.WRITE_ONLY, result_np.nbytes)

    ssim = 0
    # with open("/ssim.cl", mode="rb") as f:
    for dim in ['.x','.y','.z']:
        program = cl.Program(context, CL_SOURCE.replace('.x', dim)).build()
        program.convert(queue, (width//tile_size, height//tile_size), None,
                        img0, img1, res_g, np.int32(tile_size),
                        np.int32(width), np.int32(tile_size * tile_size),
                        np.float32(c_1), np.float32(c_2))

        cl.enqueue_copy(queue, result_np, res_g)

        # Check on CPU with Numpy:
        print(result_np)
        ssim += result_np.sum()
    # x 2.94036
    # 0.98011999

    return ssim

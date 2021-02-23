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

    source = '''//CL//
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
            /* SPLIT CODE HERE */
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
        
                // Calculate sum of the two images variances
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
            /* SPLIT CODE HERE */
        }'''
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

def get_ssim_sum(image_0, image_1, tile_size: int, pixel_len: int, width: int, height: int, c_1: float, c_2: float) -> float:
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

//CL//

__kernel void convert(
    __read_only image2d_t img0, 
    __read_only image2d_t img1, 
    __global float *res_g,
    __const int tile_size,
    __const int width,
    __const int height,
    __const float pixel_len,
    const float c_1,
    const float c_2)
{

    // Do nothing if outside of image size
    if(get_global_id(0) < (width * height)){


        // Allocate (shared) local memory
        __local uint4 pixel_sums[2][BLOCK_SIZE];
        __local uint4 covariance[BLOCK_SIZE];
        {
            // Create sampler and image position for pixel access
            const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
            // Unique pixel position for each thread
            // First part is zig zag (down right) through image; Second part cuts off at image border to next tile row
            int2 pos = (int2) ((get_global_id(0) / tile_size) % (width * tile_size),
                       (get_global_id(0) % tile_size) + tile_size * (get_global_id(0) / (tile_size * width)));

            // Each thread copies corresponding pixel to local memory
            pixel_sums[0][get_local_id(0)] = read_imageui(img0, sampler, pos);
            pixel_sums[1][get_local_id(0)] = read_imageui(img1, sampler, pos);
            covariance[get_local_id(0)] = pixel_sums[0][get_local_id(0)] * pixel_sums[1][get_local_id(0)];

        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduction for tiles
        // Stop when both operands have a different tile id
        int index = get_local_id(0) & 1; // 1 if odd, 0 if even
        int local_id_2 = (get_local_id(0) - index); // Make all Ids even
        for (int j = 1; ((local_id_2 + j) < BLOCK_SIZE) &&(j < pixel_len); j *= 2)
        {
            if((local_id_2 % (int)pixel_len) + j < pixel_len)
            {
                // Don't add two times
                pixel_sums[index][local_id_2] += pixel_sums[index][local_id_2 + j];
                if(index)
                {
                    covariance[local_id_2] += covariance[local_id_2  + j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // Add odd starting pixels
        if(!((local_id_2 - 1) % (int)pixel_len))
        {
            pixel_sums[index][local_id_2] += pixel_sums[index][local_id_2 - 1];
            if(index)
            {
                covariance[local_id_2 ] += covariance[local_id_2 - 1];
            }
        }

        if(get_global_id(0) < width) res_g[get_global_id(0)] = pixel_sums[0][get_local_id(0)].x;


    covariance[get_local_id(0)] = (covariance[get_local_id(0)] - pixel_sums[0] * pixel_sums[1] / pixel_len) / pixel_len;
    /*if(get_global_id(0)==0)
    {
        uint4 pix0;
        uint4 pix1;
        int pixel_sum_0 = 0;
        int pixel_sum_1 = 0;
        float covariance = 0;
        float variance_0 = 0;
        float variance_1 = 0;





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
    }*/
    }
}
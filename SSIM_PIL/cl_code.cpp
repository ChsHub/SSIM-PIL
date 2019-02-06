//CL//

// __global float image_0;


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
    int2 pos = (int2)(get_global_id(1) * 2, get_global_id(2) * 2); // Unique pixel position for each thread

    __local uint4 pixel_0[pixel_len];
    __local uint4 pixel_1[pixel_len];

    //pixel_0[(pos.y * width + pos.x) % width] = read_imageui(img0, sampler, pos);
    //pixel_1[pos.y * width + pos.x] = read_imageui(img1, sampler, pos);

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

        // Reduction in X-direction
        for(int x=0; x < tile_size; x += x * 2)
        {
            pixel_0[pos. y * width + pos. x] =  read_imageui(img0, sampler, pos + (int2)(x, y)) +
                                                read_imageui(img0, sampler, (int2)(x, y));
        }

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
//CL//
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
}
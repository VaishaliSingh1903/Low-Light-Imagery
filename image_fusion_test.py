import numpy as np
import imageio
import pyopencl as cl 
import matplotlib.pyplot as plt

k = np.array([0.05, 0.25, 0.40, 0.25, 0.05], dtype=np.float32)
K = k.reshape(-1,1) * k
print(K)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, 
                properties=cl.command_queue_properties.PROFILING_ENABLE)


convolution_prog = """
    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE
                                    | CLK_FILTER_NEAREST;

    __kernel void convolve_image_rgb(__read_only image2d_t src, 
                              __constant float *kernelVal, 
                              __constant int *kernel_size,
                              __write_only image2d_t dest) 
    {

            int r = get_global_id(0);
            int c = get_global_id(1);

            float4 dest_p;
            //dest_p.xyzw = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
            int kernel_idx = 0;

            for(int i=0; i< kernel_size[0]; i++) {
                for(int j=0; j< kernel_size[0]; j++) {
                    float4 pix = read_imagef(src, sampler, (int2)(r+i, c+j));
                    //dest_p.x += (pix.x * (kernelVal[kernel_idx]));
                    //dest_p.y += (pix.y * (kernelVal[kernel_idx]));
                    //dest_p.z += (pix.z * (kernelVal[kernel_idx]));
                    //dest_p.w = 1.0f;
                    dest_p.xyzw += (pix.xyzw * (kernelVal[kernel_idx]));
                    dest_p.w = 1.0f;
                    kernel_idx++;
                }
            }

            write_imagef(dest, (int2)(r+(kernel_size[0]/2), c+(kernel_size[0]/2)), dest_p);

    }

    __kernel void convolve_image_r(__read_only image2d_t src, 
                              __constant float *kernelVal, 
                              __constant int *kernel_size,
                              __write_only image2d_t dest) 
    {

            int r = get_global_id(0);
            int c = get_global_id(1);

            float dest_p = 0;
            int kernel_idx = 0;

            for(int i=0; i< kernel_size[0]; i++) {
                for(int j=0; j< kernel_size[0]; j++) {
                    float4 pix = read_imagef(src, sampler, (int2)(r+i, c+j));
                    dest_p += pix.x * (kernelVal[kernel_idx]);
                    kernel_idx++;
                }
            }

            write_imagef(dest, (int2)(r+kernel_size[0]/2, c+kernel_size[0]/2), dest_p);

    }
"""


def get_sobel_kernel(axis="x", ksize=3):
    kernel = None
    g1 = np.zeros((ksize,))

    val = 1
    for i in range(0, ksize):
        g1[i] = val
        if i < ksize//2:
            val += 1
        else:
            val -= 1 
        
    g2 = np.zeros((ksize,))
    val = -(ksize//2)
    for i in range(0, ksize):
        g2[i] = val
        val += 1

    if axis=="x":
        g1 = g1.reshape(-1, 1)
        kernel = g1 * g2
    else:
        g2 = g2.reshape(-1, 1)
        kernel = g2 * g1

    return kernel.ravel().astype(np.float32)



def get_convolved_image_rgb(queue, im, kernel, ksize=5):

    kernel = kernel.astype(np.float32)
    kernel_size = np.array([ksize], dtype=np.int32)
    kernel_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)
    kernel_size_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel_size)

    print('kernel type:', kernel.dtype)
    print('image type:', im.dtype)

    h = im.shape[0]
    w = im.shape[1]

    print('im shape:', im.shape)
    print('im.shape[2]:', im.shape[2])
    im = np.dstack(( im, np.ones(im.shape[:-1]) ))
    print('im shape:', im.shape)
    print('im.shape[2]:', im.shape[2])
    im_buf = cl.image_from_array(ctx, im.astype(np.float32), im.shape[2])
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    dest_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))
    

    prg = cl.Program(ctx, convolution_prog).build()

    print('calling convolve_image_rgb')
    exec_evt = prg.convolve_image_rgb(queue, (w, h), None, im_buf, kernel_buf, kernel_size_buf, dest_buf)
    exec_evt.wait()
    elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print("Execution time for convolve image: %g s" % elapsed)

    dest = np.empty_like(im)
    cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
    dest = dest[:, :, :-1]
    return dest



def get_convolved_image_r(queue, im, kernel, ksize=5):
        
    kernel = kernel.astype(np.float32)
    kernel_size = np.array([ksize], dtype=np.int32)
    kernel_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)
    kernel_size_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel_size)

    print('kernel type:', kernel.dtype)
    print('image type:', im.dtype)

    h = im.shape[0]
    w = im.shape[1]

    im_buf = cl.image_from_array(ctx, im.astype(np.float32), 1)
    fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    dest_buf = cl.Image(ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

    prg = cl.Program(ctx, convolution_prog).build()

    print('calling convolve_image_r')
    exec_evt = prg.convolve_image_r(queue, (w, h), None, im_buf, kernel_buf, kernel_size_buf, dest_buf)
    exec_evt.wait()
    elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
    print("Execution time for convolve image: %g s" % elapsed)

    dest = np.empty_like(im)
    cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
    return dest


def downsample(im):
    ksize = K.shape[0]
    K_ = K.ravel()

    # TODO: make convolution half-symmetric boundary extension

    # im_ = get_convolved_image(queue, im, get_sobel_kernel(ksize=3), 3)
    im_ = get_convolved_image(queue, im, K_, ksize)
    print('im shape:', im_.shape)
    plt.imshow(im_)
    plt.show()

    sobel1 = np.array([[-1, -2, -1], [0, 0,0], [1,2,1]])
    ims1 = get_convolved_image(queue, im, sobel1.ravel(), 3)
    print('Sobel:', ims1[0:5, 0:5])
    plt.imshow(ims1)
    plt.show()

    if len(im_.shape) == 2:
        imm = np.empty(( (im.shape[0]//2)-1, (im.shape[1]//2)-1  ))
    else:
        imm = np.empty(( (im.shape[0]//2)-1, (im.shape[1]//2)-1, im_.shape[2] ))
    
    grid = np.indices(( (im.shape[0]//2)-1, (im.shape[1]//2) -1 ))
    imm[grid[0].ravel(), grid[1].ravel()] = im_[grid[0].ravel()*2, grid[1].ravel()*2]
    return imm


def upsample(im, oddh, oddw):
    pass

def collapse(lp):
    pass

def calc_gaussian_pyramid(im):
    pass

def calc_laplacian_pyramid(im):
    pass


orig_im = imageio.imread('images/original.bmp')
orig_im = (orig_im/255.0).astype(np.float32)
under_ex = imageio.imread('images/under_exposed.bmp')
under_ex = (under_ex/255.0).astype(np.float32)
over_ex = imageio.imread('images/over_exposed.bmp')
over_ex = (over_ex/255.0).astype(np.float32)

flags = None
objs = None
get_supported_image_formats(ctx, cl.mem_flags.)

sh_time = 2
plt.imshow(orig_im)
plt.show(block=False)
plt.pause(sh_time)
plt.close()

plt.imshow(under_ex)
plt.show(block=False)
plt.pause(sh_time)
plt.close()

plt.imshow(over_ex)
plt.show(block=False)
plt.pause(sh_time)
plt.close()


print(under_ex[0:5, 0:5])
und_ = under_ex[:, :, 0]
orig_downsampled = downsample(und_)
orig_downsampled = np.dstack(( orig_downsampled, orig_downsampled, orig_downsampled ))
# orig_downsampled = downsample(under_ex)
print('downsampled shape:', orig_downsampled.shape)
print(orig_downsampled[0:5, 0:5])
# print(orig_downsampled==np.zeros(orig_downsampled.shape))
plt.imshow((orig_downsampled*255).astype("uint8"))
plt.show()
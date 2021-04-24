import imageio
import pyopencl as cl 
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# os.environ["PYOPENCL_CTX"]='1'
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, 
                properties=cl.command_queue_properties.PROFILING_ENABLE)



gray_image_prog = """
        __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                                    CLK_ADDRESS_CLAMP_TO_EDGE |
                                    CLK_FILTER_NEAREST;
        __kernel void gray_image(__read_only image2d_t src, __write_only image2d_t dest) {
             
            int x = get_global_id(0);
            int y = get_global_id(1);
            int2 coord = (int2)(x, y);
            
            uint4 color = read_imageui(src, sampler, coord);
            uint gray = 0.2126*color.x + 0.7152*color.y + 0.0722*color.z;
            write_imageui(dest, coord, (uint)(gray));

             //uint4 pixel = read_imageui(src, sampler, coord);
             //float4 color = convert_float4(pixel) / 255;
             //color.xyz = 0.2126*color.x + 0.7152*color.y + 0.0722*color.z;
             //pixel = convert_uint4_rte(color * 255);
             //write_imageui(dest, coord, (uint)pixel.x);

        }
    """

illumination_map_prog = """
     __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                                    CLK_ADDRESS_CLAMP_TO_EDGE |
                                    CLK_FILTER_NEAREST;
        __kernel void illumination_map(__read_only image2d_t src, __write_only image2d_t dest) {
             
            int x = get_global_id(0);
            int y = get_global_id(1);
            int2 coord = (int2)(x, y);
            
            uint4 color = read_imageui(src, sampler, coord);
            uint gray = max(color.x, color.y);
            gray = max(gray, color.z);
            write_imageui(dest, coord, (uint)(gray));

        }
"""

invert_image_prog = """
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                                    CLK_ADDRESS_CLAMP_TO_EDGE |
                                    CLK_FILTER_NEAREST;
        __kernel void invert_image(__read_only image2d_t src, __write_only image2d_t dest) {
             
            int x = get_global_id(0);
            int y = get_global_id(1);
            int2 coord = (int2)(x, y);
            
            uint4 color = read_imageui(src, sampler, coord);
            uint4 pixel = (uint4)(255-color.x, 255-color.y, 255-color.z, 255);
            write_imageui(dest, coord, pixel);

        }
"""

invert_image_normed_prog = """
    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | 
                                    CLK_FILTER_NEAREST;
    __kernel void invert_image_normed(__read_only image2d_t src, __write_only image2d_t dest) {
        int x = get_global_id(0);
        int y = get_global_id(1);

        float4 color = read_imagef(src, sampler, (int2)(x, y));
        float4 pixel = (float4)(1-color.x, 1-color.y, 1-color.z, 1);
        write_imagef(dest, (int2)(x, y), pixel);
    }
"""

convolve_prog = """
    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE
                                    | CLK_FILTER_NEAREST;

    __kernel void convolve_image(__read_only image2d_t src, __write_only image2d_t dest) {
        int r = get_global_id(0);
        int c = get_global_id(1);
        uint4 pixel = read_imageui(src, sampler, (int2)(r, c));
        write_imageui(dest, (int2)(r, c), (uint)pixel.x);
    } 
"""

sobel_prog = """
    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE
                                    | CLK_FILTER_NEAREST;

    __kernel void sobel_image(__read_only image2d_t src, 
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


class GPUImage(object):

    def __init__(self, path, ctx):
        self.ctx = ctx
        im = imageio.imread(path)
        im = np.array(im).astype(np.uint8)
        # im = im[:3, :2, :]
        
        alpha_chan = np.ones((im.shape[0], im.shape[1], 1), dtype=np.uint8)*255
        self.im = np.dstack(( im, alpha_chan ))

        # alpha_chan = np.zeros((im.shape[0], im.shape[1], 1), dtype=np.uint8)
        # im = np.dstack(( im, alpha_chan))
        # im = np.asarray(im)
        # im = np.reshape(im, self.im.shape)
        self.im_buf = cl.image_from_array(ctx, self.im, 4)
        im_float = (self.im/255.).astype(np.float32)
        self.im_buf_normed = cl.image_from_array(ctx, (self.im/255).astype(np.float32), 4)
        self.illum_map_buf = None


    def get_gray_image(self, queue):
        h = self.im.shape[0]
        w = self.im.shape[1]

        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        prg = cl.Program(self.ctx, gray_image_prog).build()

        exec_evt = prg.gray_image(queue, (w, h), None, self.im_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

        print("Execution time of gray image: %g s" % elapsed)

        dest = np.empty((self.im.shape[0], self.im.shape[1])).astype(np.uint8)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
        
        return dest


    def get_illumination_map(self, queue):
        h = self.im.shape[0]
        w = self.im.shape[1]

        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        prg = cl.Program(self.ctx, illumination_map_prog).build()

        exec_evt = prg.illumination_map(queue, (w, h), None, self.im_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

        print("Execution time of illumination map: %g s" % elapsed)

        dest = np.empty((self.im.shape[0], self.im.shape[1])).astype(np.uint8)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
        self.illum_map_buf = cl.image_from_array(self.ctx, dest, 1)
        return dest

    def get_inverted_image(self, queue):
        h = self.im.shape[0]
        w = self.im.shape[1]

        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        prg = cl.Program(self.ctx, invert_image_prog).build()

        exec_evt = prg.invert_image(queue, (w, h), None, self.im_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

        print("Execution time of invert image: %g s" % elapsed)

        dest = np.empty_like(self.im)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
        return dest

    def get_inverted_image_normed(self, queue):
        h = self.im.shape[0]
        w = self.im.shape[1]

        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        prg = cl.Program(self.ctx, invert_image_normed_prog).build()

        exec_evt = prg.invert_image_normed(queue, (w, h), None, self.im_buf_normed, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

        print("Execution time of invert image normalized: %g s" % elapsed)

        dest = np.empty(self.im.shape).astype(np.float32)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
        return dest

    def get_sobel_kernel(self, axis="x", ksize=3):
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
        


    def convolve_image(self, queue, type="Sobel", axis="x", ksize=3):
        kernel = None
        if type=="Sobel":
            kernel = self.get_sobel_kernel(axis, ksize)

        h = self.im.shape[0]
        w = self.im.shape[1]

        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        prg = cl.Program(self.ctx, convolve_prog).build()

        exec_evt = prg.convolve_image(queue, (w, h), None, self.illum_map_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

        print("Execution time of convolve image: %g s" % elapsed)

        dest = np.empty((h, w)).astype(np.uint8)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
        return dest


    def get_sobel_image(self, queue, axis="x", ksize=3):
        
        kernel = self.get_sobel_kernel(axis=axis, ksize=ksize)
        print(f'axis:{axis}, kernel:', kernel)
        
        kernel_size = np.array([ksize], dtype=np.int32)
        kernel_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)
        kernel_size_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel_size)

        h = self.im.shape[0]
        w = self.im.shape[1]

        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        prg = cl.Program(self.ctx, sobel_prog).build()

        exec_evt = prg.sobel_image(queue, (w, h), None, self.im_buf_normed, kernel_buf, kernel_size_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
        print("Execution time for sobel image: %g s" % elapsed)

        dest = np.empty((h, w)).astype(np.float32)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
        return dest

                




# gpm = GPUImage('imageio:chelsea.png', ctx)
gpm = GPUImage('images/original.bmp', ctx)
# gpm = GPUImage('jsinstability.jpeg', ctx)
sh_time = 1

print(gpm.im.shape)
print(type(gpm.im))
print(type(gpm.im_buf))

gp_gry = gpm.get_gray_image(queue)
print('gray:', gp_gry.shape)

black = np.empty((5, 5, 3))
print('black type:', black.dtype)
plt.imshow(black)
plt.show(block=False)
plt.pause(sh_time)
plt.close()

t1 = time.time()
red = gpm.im[:, :, 0]
green = gpm.im[:, :, 1]
blue = gpm.im[:, :, 2]
gry = 0.2126*red + 0.7152*green + 0.0722*blue;
gry = gry.astype(np.uint8)
t2 = time.time()
print('Without opencl:', (t2-t1), "s")

plt.imshow(gpm.im)
plt.title('Chelsea Normal')
plt.show(block=False)
plt.pause(sh_time)
plt.close()

# gp_gry[:, :, -1] = 255
plt.imshow(gp_gry, cmap="gray")
plt.title('Gray Image')
plt.show(block=False)
plt.pause(sh_time)
plt.close()

illum_map = gpm.get_illumination_map(queue)
plt.imshow(illum_map, cmap="gray")
plt.title('Illumination Map')
plt.show(block=False)
plt.pause(sh_time)
plt.close()

inverted = gpm.get_inverted_image(queue)
plt.imshow(inverted)
plt.title('Inverted Image')
plt.show(block=False)
plt.pause(sh_time)
plt.close()

inverted_normed = gpm.get_inverted_image_normed(queue)
print('inverted normed[0, 0] :', inverted_normed[0, 0])
plt.imshow(inverted_normed)
plt.title('Inverted Image normalized')
plt.show(block=False)
plt.pause(sh_time)
plt.close()

conv_img = gpm.convolve_image(queue)
plt.imshow(conv_img, cmap='gray')
plt.title('Convolved Image')
plt.show(block=False)
plt.pause(sh_time)
plt.close()

sobel_img_x = gpm.get_sobel_image(queue, axis="x", ksize=3)
# print('sobel shape:', sobel_img_x.shape)
# # print(sobel_img[0])
# plt.imshow(sobel_img_x, cmap='gray')
# plt.title('Sobel Image X')
# plt.show(block=False)
# plt.pause(sh_time)
# plt.close()

sobel_img_y = gpm.get_sobel_image(queue, axis="y", ksize=3)
# print('sobel shape:', sobel_img_y.shape)
# # print(sobel_img[0])
# plt.imshow(sobel_img_y, cmap='gray')
# plt.title('Sobel Image Y')
# plt.show(block=False)
# plt.pause(sh_time)
# plt.close()

sobel = np.empty((sobel_img_x.shape[0], sobel_img_y.shape[1]+sobel_img_x.shape[1]))
sobel[0:sobel_img_x.shape[0], 0:sobel_img_x.shape[1]] = sobel_img_x
sobel[0:sobel_img_y.shape[0], sobel_img_x.shape[1]:sobel_img_x.shape[1]+sobel_img_y.shape[1]] = sobel_img_y
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Image')
plt.show()


print(np.allclose(sobel_img_x, sobel_img_y))
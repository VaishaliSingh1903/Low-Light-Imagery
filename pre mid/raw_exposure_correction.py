import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import imageio 
import time
import os
import cv2



def get_illumination_map(im):
    return np.max(im, axis=2)


######################### Get Kernels ###################################

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

    if axis=="y":
        g1 = g1.reshape(-1, 1)
        kernel = g1 * g2
    else:
        g2 = g2.reshape(-1, 1)
        kernel = g2 * g1

    return kernel.astype(np.float32)



def get_gaussian_kernel(order=15, sigma=3):
    grid = np.indices((order, order))
    grid[0] = (grid[0]-(order//2))**2
    grid[1] = (grid[1]-(order//2))**2

    kernel = grid[0] + grid[1]
    kernel = np.exp(-kernel/(2* (sigma**2)))
    return kernel


def get_laplacian_kernel(neighbourhood=4, with_smoothing=False):
    if with_smoothing:
        gk = get_gaussian_kernel(order=3)
    lap_kernel = np.full((3, 3), -1)
    lap_kernel[1, 1] = neighbourhood
    if neighbourhood==4:
        lap_kernel[0, 0] = 0
        lap_kernel[0, 2] = 0
        lap_kernel[2, 0] = 0
        lap_kernel[2, 2] = 0
    if with_smoothing:
        return np.multiply(gk, lap_kernel)
    return lap_kernel

def get_burt_adelson_kernel():
    k = np.array([0.05, 0.25, 0.4, 0.25, 0.5])
    kt = k.reshape(-1, 1)
    return kt * k

##################################################################################################
def get_gradient_image(im, axis="x", ksize=3):
    kernel = get_sobel_kernel(axis=axis, ksize=ksize)
    # print('get_gradient_image==im shape:', im.shape, ', kernel shape:', kernel.shape)
    im_ = im.copy()
    if im_.ndim>2:
        for i in range(im_.shape[2]):
            im_[:, :, i] = convolve2d(im_[:, :, i], kernel, boundary='symm', mode='same')
    else:
        im_ = convolve2d(im_, kernel, boundary='symm', mode='same')
    return im_


def get_weight_matrix(illum_map, ksize=3, eps=1e-3):
    gaussian_kernel = get_gaussian_kernel()
    grad_x = get_gradient_image(illum_map, axis="x", ksize=ksize)
    grad_y = get_gradient_image(illum_map, axis="y", ksize=ksize)
    T_numerator = convolve2d(np.ones(illum_map.shape), gaussian_kernel, boundary='symm', mode='same')
    T_denom_x = np.abs(convolve2d(grad_x, gaussian_kernel, boundary='symm', mode='same'))+eps
    T_denom_y = np.abs(convolve2d(grad_y, gaussian_kernel, boundary='symm', mode='same'))+eps
    T_x, T_y = T_numerator/T_denom_x, T_numerator/T_denom_y
    w_x = T_x / (np.abs(grad_x)+eps)
    w_y = T_y / (np.abs(grad_y)+eps)
    return w_x, w_y

def image_inverse(im):
    return np.linalg.pinv(im)


################################# Mertens Image Fusion Methods ################################
def get_contrast(im):
    im = np.mean(im, axis=2)
    lap_kernel = get_laplacian_kernel()
    cntrst_im = np.abs(convolve2d(im, lap_kernel, boundary='symm', mode='same'))
    return cntrst_im

def get_saturation(im):
    im_mean = np.mean(im, axis=2)
    variation = np.square(im - im_mean[:, :, np.newaxis])
    sat = np.mean(variation, axis=2)
    return np.sqrt(sat)

def get_well_exposedness(im, sigma=0.2):
    dev = np.square(im - 0.5)
    dev = (-1*dev)/(2* (sigma**2))
    dev = np.exp(np.sum(dev, axis=2))
    return dev 

def get_fusion_weights(im, wc=1.0, ws=1.0, we=1.0):
    contrast_im = get_contrast(im)
    saturation_im = get_saturation(im)
    well_exposedness_im = get_well_exposedness(im)
    contrast_im = np.float_power(contrast_im, wc)
    saturation_im = np.float_power(saturation_im, ws)
    well_exposedness_im = np.float_power(well_exposedness_im, we)
    return np.multiply(contrast_im, np.multiply(saturation_im, well_exposedness_im))

def normalize_fusion_weights(im_list, axis=0):
    # print(im_list)
    im_list_sum = np.sum(im_list, axis=axis)
    im_list_sum = np.nan_to_num(im_list_sum, nan=1.0)
    im_list_sum[im_list_sum < 1e-4] = 1.0 
    im_list = im_list / im_list_sum
    return im_list

def half_symmetric_pad(im, appendH=1, appendW=1):
    # appendH = 1
    up_pad = np.tile(im[0], appendH).reshape(appendH, im.shape[1])
    down_pad = np.tile(im[-1], appendH).reshape(appendH, im.shape[1])
    im_pad = np.vstack(( up_pad, im, down_pad ))

    # appendW = (W_hat - W)//2
    # appendW = 1
    left_pad = np.tile(im_pad[:, 0], appendW).reshape(appendW, im_pad.shape[0])
    left_pad = left_pad.T 
    right_pad = np.tile(im_pad[:, -1], appendW).reshape(appendW, im_pad.shape[0])
    right_pad = right_pad.T

    im_pad = np.hstack((left_pad, im_pad, right_pad))
    return im_pad

def downsample_img2d(im):
    K = get_burt_adelson_kernel()
    # print('K :', K.shape)
    im_hat = convolve2d(im, K, boundary='symm', mode='same')
    (H, W) = im.shape[:2]
    # if (H//2)-1 < 0  or  (W//2)-1 < 0:
    #     im = half_symmetric_pad(im)
    #     (H, W) = im.shape[:2]
    v = np.zeros(( (H-1)//2, (W-1)//2 ))
    grid = np.indices(v.shape)
    grid_x = grid[0].ravel()
    grid_y = grid[1].ravel()
    v[grid_x, grid_y] = im_hat[grid_x*2, grid_y*2]
    return v


def upsample_img2d(im, oddh=2, oddw=2):
    K = get_burt_adelson_kernel()
    (H, W) = im.shape
    H_hat, W_hat = 2*(H+2), 2*(W+2)
    H_ups, W_ups = (2*H)+oddh, (2*W)+oddw
    im_pad_up = np.zeros((H_hat, W_hat))

    # increase size of u_pad by replicating first and last rows and columns
    # appendH = (H_hat - H)//2
    appendH = 1
    up_pad = np.tile(im[0], appendH).reshape(appendH, im.shape[1])
    down_pad = np.tile(im[-1], appendH).reshape(appendH, im.shape[1])
    im_pad = np.vstack(( up_pad, im, down_pad ))

    # appendW = (W_hat - W)//2
    appendW = 1
    left_pad = np.tile(im_pad[:, 0], appendW).reshape(appendW, im_pad.shape[0])
    left_pad = left_pad.T 
    right_pad = np.tile(im_pad[:, -1], appendW).reshape(appendW, im_pad.shape[0])
    right_pad = right_pad.T

    im_pad = np.hstack((left_pad, im_pad, right_pad))
    # plt.imshow(im_pad)
    # plt.title('IM pad')
    # plt.show()

    grid = np.indices((H+1, W+1))
    grid_x = grid[0].ravel()
    grid_y = grid[1].ravel()

    im_pad_up[grid_x*2, grid_y*2] = 4 * im_pad[grid_x, grid_y]

    # plt.imshow(im_pad[grid_x, grid_y].reshape(grid[0].shape))
    # plt.show()

    im_up = convolve2d(im_pad_up, K, boundary='symm', mode='same')
    im_up = im_up[2:, 2:]
    if oddh-2<0:
      im_up = im_up[:oddh-2]
    if oddw-2<0:
      im_up = im_up[:, :oddw-2]
    # im_up = im_up[:oddh-2, :oddw-2]
    return im_up


def downsample_img3d(im, oddh=2, oddw=2):
    if im.ndim < 3:
        return downsample_img2d(im)
    im_down= downsample_img2d(im[:, :, 0])

    downsampled_img = np.empty((im_down.shape[0], im_down.shape[1], im.shape[2]))-1
    downsampled_img[:, :, 0] = im_down
    for i in range(1, im.shape[2]):
        downsampled_img[:, :, i] = downsample_img2d(im[:, :, i])
    return downsampled_img

def upsample_img3d(im, oddh=2, oddw=2):
    if im.ndim < 3:
        return upsample_img2d(im, oddh=oddh, oddw=oddw)
    im_up = upsample_img2d(im[:, :, 0], oddh=oddh, oddw=oddw)
    upsampled_img = np.empty((im_up.shape[0], im_up.shape[1], im.shape[2]))
    upsampled_img[:, :, 0] = im_up
    for i in range(1, im.shape[2]):
        upsampled_img[:, :, i] = upsample_img2d(im[:, :, i], oddh=oddh, oddw=oddw)
    return upsampled_img


def get_gaussian_pyramid(im):
    gpyr = list()
    lmax = int(np.log(min(im.shape[0], im.shape[1]))/np.log(2))-1
    
    gpyr.append(im)
    
    for l in range(lmax):
        im = downsample_img3d(im)
        gpyr.append(im)
    return gpyr

def get_laplacian_pyramid(im):
    gpyr = get_gaussian_pyramid(im)
    lpyr = list()
    lmax = int(np.log(min(im.shape[0], im.shape[1]))/np.log(2))-1

    for i in range(lmax):
        oddh = gpyr[i].shape[0] - (2*gpyr[i+1].shape[0])
        oddw = gpyr[i].shape[1] - (2*gpyr[i+1].shape[1])
        lpyr.append(gpyr[i]-upsample_img3d(gpyr[i+1], oddh=oddh, oddw=oddw))

    lpyr.append(gpyr[lmax])
    return lpyr

def collapse_pyramid(lpyr):
    lrange = list(reversed(list(range(len(lpyr)-1))))
    for l in lrange:
        oddh = lpyr[l].shape[0] - (2*lpyr[l+1].shape[0])
        oddw = lpyr[l].shape[1] - (2*lpyr[l+1].shape[1])
        lpyr[l] = lpyr[l] + upsample_img3d(lpyr[l+1], oddh=oddh, oddw=oddw)
    return lpyr[0]

def merge_mertens(orig_im, under_ex, over_ex, wc=1, ws=1, we=1):
    weights = list()
    weights.append(get_fusion_weights(orig_im, wc=wc, ws=ws, we=we))
    weights.append(get_fusion_weights(under_ex, wc=wc, ws=ws, we=we))
    weights.append(get_fusion_weights(over_ex, wc=wc, ws=ws, we=we))
    weights = np.array(weights)
    weights = weights + 1e-12
    print('weights shape:', weights.shape)
    weights = normalize_fusion_weights(weights, axis=0)

    im_list = [orig_im, under_ex, over_ex]
    out_lpyr = get_laplacian_pyramid(orig_im)
    for l in out_lpyr:
        l[:, :, :] = 0
    
    for i in range(weights.shape[0]):
        weight_gpyr = get_gaussian_pyramid(weights[i])
        img_lpyr = get_laplacian_pyramid(im_list[i])
        collapse_pyramid(img_lpyr)
        for l in range(len(img_lpyr)):
            out_lpyr[l] = out_lpyr[l] + np.multiply(img_lpyr[l], weight_gpyr[l][:, :, np.newaxis])

    out_img = out_lpyr[0]
    # out = collapse_pyramid(out_lpyr)
    return out_img

def get_neighbours(h, w):
    grid = np.indices((h, w))
    leftn = [grid[0, :, :-1], grid[1, :, :-1]]
    rightn = [grid[0, :, 1:], grid[1, :, 1:]]
    topn = [grid[0, :-1, :], grid[1, :-1, :]]
    bottomn = [grid[0, 1:, :], grid[1, 1:, :]]
    return leftn, rightn, topn, bottomn


def get_sparse_neighbor(p, n, m):
    i, j = p // m, p % m
    d = dict()
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)
    return d


def construct_F(illum_map, wx, wy):
    (h, w) = illum_map.shape
    # leftn, rightn, topn, bottomn = get_neighbours(h, w)

    # grid = np.indices((h, w))
    start = time.time()
    row, column, data = [], [], []
    for p in range(h * w):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, h, w).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(h * w, h * w))
    print('Time taken for construction of F:', time.time()-start, ' s')
    return F

def get_expose_corrected(img, gamma=0.6, balance_param=0.15):
    illum_map = get_illumination_map(img)
    wx, wy = get_weight_matrix(illum_map)
    flat_illum_map = illum_map.copy().flatten()
    (h,w) = illum_map.shape

    F = construct_F(illum_map, wx, wy)
    I = diags([np.ones(h*w)], [0])
    A = I + (balance_param*F)
    
    start = time.time()
    illum_map_corrected = spsolve(csr_matrix(A), flat_illum_map, permc_spec=None, use_umfpack=True)
    illum_map_corrected = illum_map_corrected.reshape((h, w))
    print('Time taken to solve Sparse Matrix: ', time.time()-start, ' s')

    illum_map_corrected = np.clip(illum_map_corrected, 1e-3, 1)**gamma
    illum_map_corrected = np.dstack(( illum_map_corrected, illum_map_corrected, illum_map_corrected ))
    img_corrected = img / illum_map_corrected
    return img_corrected



def correct_exposure(im, gamma=0.6, balance_param=0.15, wc=1, ws=1, we=1):
    im = (im/255.0).astype(np.float32)
    inv_im = 1 - im
    inv_corrected = get_expose_corrected(inv_im, gamma, balance_param)
    im_corrected = get_expose_corrected(im, gamma, balance_param)

    # out_img = merge_mertens(im, im_corrected, inv_corrected)
    
    im_corrected = np.clip(im_corrected*255, 0, 255).astype("uint8")
    inv_corrected = np.clip(inv_corrected*255, 0, 255).astype("uint8")
    im = np.clip(im*255, 0, 255).astype("uint8")

    mm = cv2.createMergeMertens(wc, ws, we)
    out_img = mm.process([im, im_corrected, inv_corrected])

    out_img = np.clip(out_img*255, 0, 255).astype("uint8")
    return im, im_corrected, inv_corrected, out_img



sh_time = 2
# im = imageio.imread('images/original.bmp')
# im = imageio.imread('imageio:chelsea.png')
# im = imageio.imread('images/person_exdark.jpg')

filename = 'toy_LOL.png'
out_folder = 'output_image'
im = imageio.imread(f'images/{filename}')

# im = imageio.imread('images/toy_LOL.png')
img, img_cr, img_inv_cr, img_out = correct_exposure(im)

plt.imshow(img)
plt.title('Original image')
plt.show(block=False)
plt.pause(sh_time)
plt.close()
imageio.imwrite(f'{out_folder}/{filename}', img)

plt.imshow(img_cr)
plt.title('Image Corrected')
plt.show(block=False)
plt.pause(sh_time)
plt.close()
imageio.imwrite(f'{out_folder}/corrected_{filename}', img_cr)

plt.imshow(img_inv_cr)
plt.title('Image Inverted Corrected')
plt.show(block=False)
plt.pause(sh_time)
plt.close()
imageio.imwrite(f'{out_folder}/inv_corrected_{filename}', img_inv_cr)

plt.imshow(img_out)
plt.title('Corrected Output')
plt.show()
imageio.imwrite(f'{out_folder}/final_{filename}', img_out)
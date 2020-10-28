import os
import glob
import imageio
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.color import gray2rgb, rgb2gray

from scipy import ndimage, signal
from imageio import imread, imsave

def quantize_rgb(img: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-means clusters for the input image in RGB space, and return
    an image where each pixel is replaced by the nearest cluster's average RGB
    value.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        An RGB image with shape H x W x 3 and dtype "uint8"
    """
    quantized_img = np.zeros_like(img)
    working_img = np.reshape(img, (img.shape[0]*img.shape[1],3))
    clustered = MiniBatchKMeans(n_clusters=k, random_state=101, batch_size=1024).fit(working_img)
    labels = clustered.predict(working_img)
    centers = clustered.cluster_centers_
    labels = np.reshape(labels, (img.shape[0], img.shape[1]))
    working_img = np.reshape(working_img, (img.shape[0],img.shape[1],img.shape[2]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cluster = labels[i,j]
            cluster_center = centers[cluster,:]
            quantized_img[i,j,:] = cluster_center

    quantized_img = quantized_img.astype('uint8')

    return quantized_img

def quantize_xray(filepath, clusters):
    """
    Compute k-means clusters for the input xray image in RGB space,
    and return an image where each pixel is replaced by the nearest 
    cluster's average RGB value.

    Inputs: 
        filepath: filepath of desired xray image
        clusters: The number of clusters to use

    Output: 
        A quantized RGB image of the xray
        Also saves 2D image in directory with same name as 
        input image with 'b' prepended
    """

    k = clusters
    img = imageio.imread(filepath)
    if img.ndim > 2:
        img = img[:,:,0]
    blur = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
    convimg = np.zeros_like(gxray)
    ndimage.convolve(input=gxray, weights = blur, mode='constant', output = convimg)
    rgbimg = gray2rgb(convimg)
    quantized_img = quantize_rgb(rgbimg, k)
    imageio.imwrite('b' + x, conv[:, :, 0])

    return quantized_img

from .utils import *
import math
from spectral import *
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from math import sin, cos
from tifffile import imsave, imread
from skimage import transform
from copy import deepcopy
import torch
from .types import SingleCellData
from typing import List
from concurrent.futures import ThreadPoolExecutor


executor = ThreadPoolExecutor(max_workers=16)


def get_wavelengths(envi_file):
    """Extract wavelengths from ENVI file
    :param envi_file: input ENVI file
    :return: wavelengths
    """
    envi = enviread(envi_file)
    wavelengths = envi.bands.centers
    return wavelengths


def percentile_stretch(img, percent=2.0, minval=0, maxval=1.0):
    """Apply percentile stretching to image. This function set image minimum as "percent"
        and image maximum as 100% - "percent" and apply linear stretching between the two values
    img: input greyscale image
    percent: lower percentage to be minimum for stretching
    """
    data = img.ravel()
    min0 = np.percentile(data, percent)
    max0 = np.percentile(data, 100-percent)
    ret = img
    ret[ret < min0] = min0
    ret[ret > max0] = max0
    ret = (ret-min0)/(max0-min0)
    return ret


def percentile_stretch_optimized(img, min_perc=0.025, max_perc=0.99,
                                 min_adj_perc=0.1, max_adj_perc=0.5,
                                 minval=0, maxval=1.0):
    """Apply optimized percentile stretching to image.
    """
    data = img.ravel()
    a = np.percentile(data, min_perc*100)
    b = np.percentile(data, max_perc*100)
    c = a - 0.1*(b-a)
    d = b + 0.5*(b-a)
    ret = img
    ret[ret < c] = c
    ret[ret > d] = d
    ret = (ret-c)/(d-c)
    return ret


def percentile_stretch_2sides(img, lower_percent=2.0, upper_percent=98.0):
    """Apply percentile stretching to image. This function set image minimum as "percent"
        and image maximum as 100% - "percent" and apply linear stretching between the two values
    img: input greyscale image
    lower_percent: lower percentage to be minimum for stretching
    uppper_percent: upper percentage to be maximum for stretching
    """
    data = img.ravel()
    min0 = np.percentile(data, lower_percent)
    max0 = np.percentile(data, upper_percent)
    ret = img
    ret[ret < min0] = min0
    ret[ret > max0] = max0
    ret = (ret-min0)/(max0-min0)
    return ret


def enviread(filename):
    """Read hyperspectral file and return ENVI object
     Args:
          filename: file name string
     Output:
          ENVI object (spectral module)
    """
    fpath, fname, fext = fileparts(filename)
    hdr_name = fpath+r"/"+fname+".hdr"
    hyp_name = filename
    img = envi.open(hdr_name, image=hyp_name)
    return img


def enviread_band(filename, band):
    """Read hyperspectral file and return ENVI object of specified bands
     Args:
          filename: file name string
     Output:
          ENVI object (spectral module)
    """
    hyp = enviread(filename)
    img_band = hyp[:,:,band]
    return img_band



def spectral_matrix(envi, mask):
    """Retrieve 2D matrix (npixels x nbands) from
        hyperspectral image envi (ny,nx,nbabds) and mask (ny,nx)
        where npixels is number of non-zero pixels in mask0
     Args:
          envi: ENVI object of (ny,nx,nbabds)
          mask: 2D array of (ny,nx)
     Output:
          2D array of (npixels x nbands) where npixels is
            number of non-zero pixels in mask0
    """
    if mask.shape != envi.shape[:2]:
        raise Exception(f"first two dimensions of two arguments should be the same! mask shape: {mask.shape}, envi shape: {envi.shape[:2]}")

    # find rows cols combinations for mask pixels
    rows, cols = np.where(mask > 0)

    # return the spectral matrix
    output = np.zeros((rows.size, envi.shape[2]))
    for i, (row, col) in enumerate(zip(rows,cols)):
        output[i,:] = envi[row, col, :]
    return output


def make_intensity_file(envi_file, out_folder=None, band=72, postfix="546nm", min_max_scale=True):
    """Create tif file of 546nm band image for input hypercube with
        same filename with different file extension
    :param: envi_file:  filename for 3D ENVI hypercube object (ny,nx,nbands)
    :param: out_folder:  output folder. If it is None, it does not save the image
    :param: band: band number. For example, band = 72 (73 with 1-base) for TruScope FPI or
                band = 24 (or 25 with 1-base)
    :return: img: output band image
    :return: out_ifilename: full filename of output image
    """
    from tifffile import imwrite
    src_efpath, efname, efext = fileparts(envi_file)
    img = enviread_band(envi_file, band)

    if min_max_scale:
        xmin = img.min()
        scale = 1.0/(img.max()-xmin)
        img = (img.astype(np.float32)-xmin)*scale

    out_ifilename = None
    if out_folder is not None:
        out_ifilename = fr"{out_folder}\{efname}_{postfix}.tif"
        imwrite(out_ifilename, img)

    return img, out_ifilename


def deconvolute_for_bacteria_segment(img, sigma=11, lower_percent=1, upper_percent=99.0):
    """Deconvolute (de-blur) and contrast-stretch bacterial intensity image for segmentation
    :param img: input 2D intensity image
    :param sigma: Gaussian filter option for sigma
    :param lower_percent: lower_bound for constrast-stretch
    :param upper_percent: upper_bound for constrast-stretch
    """
    from skimage.filters import gaussian
    bg = gaussian(img, sigma=sigma, mode="reflect", truncate=2)

    # remove the background
    bg_del = img - bg
    bg_del[bg_del < 0] = 0
    bg_del = percentile_stretch_2sides(bg_del,
                                       lower_percent=lower_percent, upper_percent=upper_percent)
    # bg_del = min_max_normalize(bg_del)
    return bg_del


def pad2resize(image, max_dim, constant_val=-1):  # to resize image
    """resize image to 64x64 by padding with constant value
    :param image: input image
    :param max_dim: first two dimension of the target image
    :param constant_val: constant value for padding
    """
    max_h, max_w = max_dim
    h, w = image.shape[:2]
    top_pad = (max_h - h) // 2
    bottom_pad = max_h - h - top_pad
    left_pad = (max_w - w) // 2
    right_pad = max_w - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    image_new = np.pad(image, padding, mode='constant', constant_values=constant_val)
    return(image_new)


def fill_padded_area(image, min_dim, constant_val=0):
    """pad image without resizing"""
    image_new = image
    max_h, max_w = image.shape[:2]
    h, w = min_dim
    top_pad = (max_h - h) // 2
    bottom_pad = top_pad + h
    left_pad = (max_w - w) // 2
    right_pad = left_pad + w
    image_new[:top_pad, :] = constant_val
    image_new[bottom_pad:, :] = constant_val
    image_new[:, :left_pad] = constant_val
    image_new[:, right_pad:] = constant_val
    return(image_new)


def find_all_rightsized_blobs(input, min_area, max_area):
    """Remove too small and too large blobs out
    :param input: input 2D black & white image
    :param min_area: minimum number of pixels for candidate blob
    :param max_area: maximum number of pixels for candidate blob
    """
    mask = input.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # loop through all blobs
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        submask = mask[y:y + h, x:x + w]
        area = cv2.countNonZero(submask)
        if area < min_area or area > max_area:
            # remove the blob
            mask = cv2.drawContours(mask, [cnt], 0, (0), -1)
        else:
            mask = cv2.drawContours(mask, [cnt], 0, (255), -1)
    return mask


def gof_fit_ellipse(coords):
    """Compute Mean absolute deviation (MAD) between blob boundary and its ellipse fit
    """
    max_gof = 1.0e+305
    try:
        fit_ellipse = cv2.fitEllipse(coords)
    except:
        return max_gof, 1e+307
    angle = fit_ellipse[2] / 180 * 3.1415926535  # angle
    center = fit_ellipse[0]  # center
    cx = center[0]
    cy = center[1]
    sz = fit_ellipse[1]  # size
    szx = sz[0]
    szy = sz[1]
    g_GOF = 0
    for i, xy in enumerate(coords):
        x = xy[0, 0]
        y = xy[0, 1]
        posx = (x - cx) * cos(-angle) - (y- cy) * sin(-angle)
        posy = (x - cx) * sin(-angle) + (y- cy) * cos(-angle)
        g_GOF += abs( posx / szx * posx / szx + posy / szy * posy / szy - 0.25)

    area = 3.141592654*szx*szy
    return g_GOF/len(coords), area


def find_all_elliptical_blobs(input, is_select_all=False, min_gof=8.0):
    """Find all elliptical blobs from input image and make mask image
    :param input: input mask image
    :param is_select_all: If true, it selects all blobs
    :param min_gof: minimum goodness of fit
    :return mask: resulting mask image
    :return removed_mask: resulting image containing removed blobs
    :return blob_count: number of blobs identified
    :return total_blobs: number of blobs in input mask

    The algorithm selects a blob as a single cell if GOF = Val2 / Val1 > min_gof
    Val1: Mean absolute deviation (MAD) between blob boundary and its ellipse fit
    Val2: 1 – |ellipse area-blob area|/(ellipse area)
    Val2 helps ignore incomplete elliptical blob (e.g., half ellipse)
    The GOF value (0≤ x ≤∞) gets higher as the fit is better
    For example, min_gof=9 implies that MAD < 1/9 (pixels) assuming
    the blob area is close to ellipse area.
    """
    mask = input.copy()
    removed_mask = np.zeros_like(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Select elliptical blobs only
    total_blobs = len(contours)
    blob_count = 0
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        submask = mask[y:y + h, x:x + w]
        area = cv2.countNonZero(submask)
        # calculate Val1 in comment above
        gof, ellipse_area = gof_fit_ellipse(cnt)
        # calculate Val2 in comment above
        if ellipse_area == 0:
            gof_2 = 0
        else:
            gof = 1 / gof  # higher gof -> better gof
            gof_2 = 1 - abs(ellipse_area - area) / ellipse_area  # higher gof -> better gof

        if is_select_all:
            mask = cv2.drawContours(mask, [cnt], 0, (255), -1)
            blob_count += 1
        else:
            # select elliptical blob based on GOF
            if gof*gof_2 < min_gof:
                # remove the blob
                mask = cv2.drawContours(mask, [cnt], 0, (0), -1)
                removed_mask = cv2.drawContours(removed_mask, [cnt], 0, (255), -1)
            else:
                mask = cv2.drawContours(mask, [cnt], 0, (255), -1)
                blob_count += 1

    return mask, removed_mask, blob_count, total_blobs


def imcrop_center(img, org_dim):
    """crop center part of the image
    :param img: input image
    :param org_dim: first two dimensions of the target image size
    """
    max_h, max_w = img.shape[:2]
    h, w = org_dim
    top_pad = (max_h - h) // 2
    left_pad = (max_w - w) // 2
    new_img = img[top_pad:(top_pad+h), left_pad:(left_pad+w)]
    return new_img



def min_max_normalize(img):
    """Normalize data with its min and max
     Args:
          img: 2D or 3D array
     Output:
          2D or 3D array
    """
    out = (img - img.min()) / (img.max() - img.min())
    return out


def extract_shape_stats(cnt, mask):
    """Extract morphological properties from contour
    :param cnt: OpenCV contour object
    :param mask: mask iamge for counting pixels of blob area
    """
    # compute area of the blob (number of pixels)
    x, y, w, h = cv2.boundingRect(cnt)
    subImg = mask[y:y + h, x:x + w]
    area = cv2.countNonZero(subImg)
    # I commented this out for MSU's small cells
    # if area < 150:
    #     raise ValueError("detected area is smaller than 150")
    # center, axis_length and orientation of ellipse
    (center, axes, orientation) = cv2.fitEllipse(cnt)
    orientation = math.radians(orientation)
    # length of MAJOR and minor axis
    majoraxis_length = max(axes)
    minoraxis_length = min(axes)
    # minor/major axes ratio
    min_max_ratio = minoraxis_length / majoraxis_length
    # eccentricity
    eccentricity = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)
    # area-equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi)
    # extent
    extent = float(area) / w / h
    # perimeter
    perimeter = cv2.arcLength(cnt, True)
    return center, area, eccentricity, majoraxis_length, minoraxis_length, \
        min_max_ratio, equi_diameter, orientation, extent, perimeter

def threaded_calculate_blob_statistics(envi, mask, cnt) -> SingleCellData:
        mask0 = np.zeros_like(mask)
        cv2.drawContours(mask0, [cnt], 0, (255), -1)
        
        x, y, w, h = cv2.boundingRect(cnt)
    
        mask_sub = mask0[y:y + h, x:x + w]
        envi_sub = envi[y:y + h, x:x + w, :]
        spec_mean = spectral_matrix(envi_sub, mask_sub).mean(axis=0)
        
        return SingleCellData(contour=cnt, mean_spectra=spec_mean)

def get_single_cell_spectra(envi_file, mask) -> List[SingleCellData]:
    """Prepares fusion data
        a. identify each blob,
        b. save into a single intensity image per blob,
        c. record its file location, its shape statistics and
            average spectrum along with label into a file.
    :param envi_file: filename for 3D ENVI hypercube object (ny,nx,nbands)
    :param mask: mask image of single cell bacteria
    :return: single_cell_results: list of SingleCellData objects
    """
    global executor
    # read hypercube
    envi = enviread(envi_file)

    # find all blobs
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(f"mask shape: {mask.shape}, envi shape {envi.shape}, contours: {len(contours)}")

    start = time.time() 
    args = [(envi, mask, cnt) for cnt in contours]
    results = list(executor.map(threaded_calculate_blob_statistics, *zip(*args)))
    threaded_time = time.time()-start
    print("threaded_runtime", threaded_time)
    return results

def mkdir(out_dir):
    """create folder if it does not exist yet"""
    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


def snv(input_data):
    """Define a new array and populate it with the corrected data by Daniel Pelliccia"""
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :], ddof=1)
    return data_snv


def hex_to_rgb(h):
    """Convert Hex color string (e.g., "#ff12e0") TO COLOR LIST OF TRIPLET (e.g., [255, 18, 239])"""
    h = h.lstrip('#')
    return list(int(h[i:i + 2], 16) for i in (0, 2, 4))

def get_intensity_image(envi_file, sigma=11):
    """Extract/save 546nm band image from hypercube and Deconvolute (de-blur) and contrast-stretch image for segmentation
    :param envi_file: input ENVI file
    :return:img_intensity: intensity image
    :return:intensity_file: intensity file name
    :return:img_input: deblurred image for segmentation
    """
    # 	• Extract 546nm band image from hypercube
    band_name = "546nm"  # band_name="546nm"
    band_num = 72  # band_num=72
    img_intensity, _ = make_intensity_file(envi_file, min_max_scale=False, band=band_num, postfix=band_name)
    normalized =  min_max_normalize(img_intensity)
    # normalized =  percentile_stretch_optimized(img_intensity, max_perc=0.999, min_perc=0.01)
  
    img_intensity = np.squeeze(img_intensity)

    # resize for model
    deblurred = cv2.resize(img_intensity, (1936, 1216), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    print(f"sigma: {sigma}")
    # 	• Deconvolute (de-blur) and contrast-stretch image for segmentation
    deblurred = deconvolute_for_bacteria_segment(deblurred, sigma=sigma, lower_percent=1, upper_percent=99.0)

    # 	• Resize it to 2048 x 1280 with -1 padding for input of neural network model
    img_height = 1280
    img_width = 2048
    img_input = pad2resize(deblurred, [img_height, img_width])
    print(f"unet input size {img_input.shape}")
    # only for debugging
    # input_file = fr"{outdir}\DL_input.tif"
    # imsave(input_file, img_input)
    return img_intensity, img_input


def find_single_cells(mask, min_gof=8.0, org_img_dim = [1216, 1936]):
    """Find single cells from the mask image, and
    make/save the mask of the single cells
    :param mask: input mask of bacterial cells from DL algorithm
    :param min_gof: minimum gof value
    :return: mask_final: mask image of single cell bacteria
    """
    # 	• Pad the mask with zero for image processing
    
    mask = fill_padded_area(mask, org_img_dim)

    # 	• Filter too small and too large blobs out
    min_area = 150
    max_area = 2250
    mask2 = find_all_rightsized_blobs(mask, min_area, max_area)

    mask_final, mask_removed, blob_count, total_blobs = find_all_elliptical_blobs(mask2,
                                                                                  is_select_all=False,
                                                                                  min_gof=min_gof)

    mask_final = imcrop_center(mask_final, org_img_dim)
    
    return mask_final





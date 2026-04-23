

#mport Augmentor
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torchvision.transforms as T
from numpy import *
import torch
from scipy import ndimage
import pandas as pd
from math import log10, sqrt
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import exposure
import scipy.signal
import cv2


import time
import glob
import os
import sys
import subprocess
import csv
from scipy.ndimage import uniform_filter, gaussian_filter
#First step, reconstructing using a given beta(noted alpha in there )
def reconstruction_fn(beta,nb_img):
    os.environ['CASTOR_CONFIG'] = "/home/user/Documents/phd/codes_and_data/castor_gitlab/config"
    castor_command = ["/home/user/Documents/phd/codes_and_data/castor_gitlab/bin/castor-recon", "-df", "/home/user/Documents/phd/codes_and_data/brain_phantom/data0_1/data0_1.cdh",
                    "-opti", "BSREM", "-dim", "160,160,1", "-vox", "2.,2.,2.","-opti-fom",
                    "-pnlt", "MRF:/home/user/Documents/phd/codes_and_data/brain_phantom/mymrf.conf",
                    "-pnlt-beta", str(beta), "-it", "30:28", "-conv", "gaussian,4,1,3.5::psf",
                    "-dout", "/home/user/Documents/phd/codes_and_data/brain_phantom/output/out_brain"]

    # Execute the CASTOR command as a subprocess
    subprocess.run(castor_command, stdout=subprocess.DEVNULL)
    root='/home/user/Documents/phd/codes_and_data/brain_phantom/output/out_brain/out_brain_it30.img'
    fo = open(root, "rb")
    img = np.fromfile(fo, dtype=np.float32).reshape((160,160))
    return img
def reconstruction_beta_gamma(beta,nb_img,gamma):
     # Modify the file_path accordingly
    file_path = "/home/user/Documents/phd/codes_and_data/generalization_data/mymrf_gamma_beta.conf"

    # Read the content of mymrf.conf
    with open(file_path, "r") as f:
        lines = f.readlines()

    count = 0

    for i, line in enumerate(lines):
        if "gammaRD" in line:
            count += 1
            
            if count == 2:
                val = gamma
                lines[i] = "gammaRD: {}.\n".format(val)

                # Write the modified content back to mymrf.conf
                with open(file_path, "w") as f:
                    f.writelines(lines)
    os.environ['CASTOR_CONFIG'] = "/home/user/Documents/phd/codes_and_data/castor_gitlab/config"
    castor_command = ["/home/user/Documents/phd/codes_and_data/castor_gitlab/bin/castor-recon", "-df", "/home/user/Documents/phd/codes_and_data/generalization_data/web3M/data0_1/data0_1.cdh",
                    "-opti", "BSREM", "-dim", "160,160,1", "-vox", "2.,2.,2.","-opti-fom",
                    "-pnlt", "MRF:/home/user/Documents/phd/codes_and_data/generalization_data/mymrf_gamma_beta.conf",
                    "-pnlt-beta", str(beta), "-it", "30:28", "-conv", "gaussian,4,1,3.5::psf",
                    "-dout", "/home/user/Documents/phd/codes_and_data/brain_phantom/output/out_brain"]

    # Execute the CASTOR command as a subprocess
    subprocess.run(castor_command, stdout=subprocess.DEVNULL)
    root='/home/user/Documents/phd/codes_and_data/brain_phantom/output/out_brain/out_brain_it30.img'
    fo = open(root, "rb")
    img = np.fromfile(fo, dtype=np.float32).reshape((160,160))
    return img
def mssim(img1, img2, alpha, beta, gamma,threshold=0.001):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2
    """
    
    # Convert to float64 to avoid floating point error and negative values in sigma1_sq or sigma2_sq
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # Convert to float64 to avoid floating point error and negative values in sigma1_sq or sigma2_sq
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
     # Create a mask where img2 > threshold
    mask = img2 > threshold
    
    # Apply mask to both images
    img1 = img1[mask]
    img2 = img2[mask]
    # Data range
    L = np.max(img2) - np.min(img2)
    
    # Parameters from Wang et al. 2004
    sigma = 1.5
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    
    # Convolve images (gaussian or uniform filter) to get mean for each patch
    # filter_args = {'win_size': 7} # backwards compatibility (value from python implementation)
    # mu1 = uniform_filter(img1,**filter_args)
    # mu2 = uniform_filter(img2,**filter_args)
    filter_args = {'sigma': sigma, 'truncate': 3.5} # 3.5 is the number of sigmas to match Wang et al. to have filter size=11
    mu1 = gaussian_filter(img1,**filter_args)
    mu2 = gaussian_filter(img2,**filter_args)
        
    # Multiply images
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    # Convolve images (gaussian or uniform filter) to get variance and covariance for each patch. Remove negative values coming from floating point errors
    # sigma1_sq = uniform_filter(img1*img1,**filter_args) - mu1_sq
    # sigma2_sq = uniform_filter(img2*img2,**filter_args) - mu2_sq
    # sigma12 = uniform_filter(img1*img2,**filter_args) - mu1_mu2
    sigma1_sq = gaussian_filter(img1*img1,**filter_args) - mu1_sq
    sigma1_sq[sigma1_sq < 0] = 0
    sigma2_sq = gaussian_filter(img2*img2,**filter_args) - mu2_sq
    sigma2_sq[sigma2_sq < 0] = 0
    sigma12 = gaussian_filter(img1*img2,**filter_args) - mu1_mu2
    
    # Compute luminance, contrast and structure for each patch
    luminance =((2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1))**alpha
    contrast=((2*np.sqrt(sigma1_sq*sigma2_sq) + C2)/(sigma1_sq + sigma2_sq + C2))**beta
    structure=((2*sigma12 + C2)/(2*np.sqrt(sigma1_sq*sigma2_sq) + C2))**gamma
    
    # Compute MSSIM
    MSSIM=np.mean(luminance*contrast*structure)
    return MSSIM


def erode_mask(mask):
   
    eroded_mask = np.zeros_like(mask)
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] and np.all(mask[i - 1:i + 2, j - 1:j + 2]):
                eroded_mask[i, j] = True
    
    return eroded_mask

def compute_overall_ar_h(ground_truth, reconstructed, threshold=8):
    
    hot_mask = ground_truth >= threshold

    # Erode the hot regions
    eroded_hot_mask = erode_mask(hot_mask)

    # Label the eroded hot regions
    labeled_hot_regions, num_features = ndimage.label(eroded_hot_mask)

    # Initialize activity recovery sum and count
    total_ar_h = 0
    count_valid_regions = 0

    # Loop through each labeled region to calculate activity recovery
    for region in range(1, num_features + 1):
        region_mask = labeled_hot_regions == region
        
        # Calculate mean intensity in the ground truth and reconstructed images
        mean_gt = np.mean(ground_truth[region_mask])
        mean_reconstructed = np.mean(reconstructed[region_mask])
        
        # Calculate activity recovery for this region
        if mean_gt > 0:  # Avoid division by zero
            ar_h = mean_reconstructed / mean_gt
            total_ar_h += ar_h
            count_valid_regions += 1

    # Calculate the overall activity recovery
    overall_ar_h = total_ar_h / count_valid_regions if count_valid_regions > 0 else 0
    return overall_ar_h

def masked_psnr(pred, img_gt, threshold=0.01):
    
    # Create a mask where img_gt > threshold
    mask = img_gt > threshold
    
    # Apply the mask to both predicted and ground truth images
    pred_masked = pred[mask]
    img_gt_masked = img_gt[mask]
    
    # Compute PSNR for the masked region
    psnr_value = psnr(img_gt_masked, pred_masked, data_range=img_gt_masked.max()-img_gt_masked.min())
    
    return psnr_value


def calculate_rbf(image_M, image_F):
    # Threshold value for the hot region
    threshold = 50

    # Convert the images to NumPy arrays
    M_array = np.array(image_M)
    F_array = np.array(image_F)

    # Calculate the average of pixels above the threshold in the hot region for image M (m0)
    m0 = np.mean(M_array[M_array < threshold])

    # Calculate the overall average of all pixels in image F (f0)
    f0 = np.mean(F_array[F_array < threshold])

    # Calculate the ratio m0/f0
    ratio = (m0-f0) / f0

    return ratio
def Psnr(beta,nb_img,gamma):
    
    """
    Computes the PSNR of the reconstructed image.

    Args:
        beta (float): Regularization parameter.

    Returns:
        float: Negative value of the PSNR.
    """
    
    # original image
    if np.isnan(beta) or np.isinf(beta):
            # Return a default value: the midpoint of the bracket
            beta = (0.001 + 0.5) / 2
    input_file_path = '/home/user/Documents/phd/codes_and_data/brain_phantom/image50_2.img'

    # Define the dimensions of the image (you need to know the dimensions of your image)
    image_shape = (160,160)
    # Step 1: Load the .img file
    with open(input_file_path, 'rb') as f:
        original_image = np.fromfile(f, dtype=np.float32).reshape(image_shape)

    # Reconstruct the image with the desired regularization strength
    reconstructed_image = reconstruction_beta_gamma(beta,nb_img,gamma)
   
    reconstructed=reconstructed_image
    original=original_image
    # Calculate the correlation coefficient
    #Psnr_c= np.corrcoef(flatten1, flatten2)[0,1]
    p = masked_psnr(reconstructed, original)
    

    #arh=(1-Psnr_c)**2
    #Psnr_c2=calculate_rbf(reconstructed_image, original_image)
    ssim=mssim(reconstructed, original, 0.5, 1, 1)
    # Return the  PSNR value
    return reconstructed,ssim,p

def process_mask_cold(image, lower_threshold, upper_threshold, max_radius_threshold, min_area_threshold):
    initial_mask = (image > lower_threshold) & (image < upper_threshold)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(initial_mask.astype(np.uint8), connectivity=8)
    processed_mask = np.zeros_like(initial_mask, dtype=bool)  # Make sure to use bool for logical operations

    # Define the centered region of interest
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_size = 45 // 2
    roi_start_x, roi_start_y = center_x - half_size, center_y - half_size
    roi_end_x, roi_end_y = center_x + half_size, center_y + half_size
    central_region_mask = np.zeros_like(image, dtype=bool)  # Use bool here
    central_region_mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = True

    # Combine the central region mask with the initial mask
    initial_mask &= central_region_mask  # Should work now without error

    # Process each component within the masked region
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            component_mask = (labels == i).astype(bool)
            if np.any(component_mask & central_region_mask):  # Check component is within the central region
                # Calculate the max distance from the centroid within the component
             
                eroded_component = cv2.erode(component_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                processed_mask |= eroded_component

    return processed_mask

    
def process_mask_hot(image, lower_threshold, upper_threshold):
    initial_mask = (image > lower_threshold) & (image < upper_threshold)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(initial_mask.astype(np.uint8), connectivity=8)
    processed_mask = np.zeros_like(initial_mask, dtype=np.uint8)
    for i in range(1, num_labels):
   
        if stats[i, cv2.CC_STAT_AREA] >= 6:
            component_mask = (labels == i).astype(np.uint8)
            eroded_component = cv2.erode(component_mask, np.ones((3, 3), np.uint8), iterations=1)
            processed_mask = processed_mask | eroded_component
            
    return processed_mask
def show_mask_overlay(ground_truth_image, hot_mask, cold_mask):
    plt.figure(figsize=(6, 6))
    
    # Display the ground truth image in reversed grayscale
    plt.imshow(ground_truth_image, cmap='gray_r')
    
    # Overlay hot mask in red with transparency
    plt.contour(hot_mask, colors='red', linewidths=1.5)

    # Overlay cold mask in blue with transparency
    plt.contour(cold_mask, colors='blue', linewidths=1.5)

    plt.title("Ground Truth with Hot (Red) and Cold (Blue) Masks")
    plt.axis("off")
    plt.show()
def compute_ar_rb(target_image, ground_truth_image, gt_original):
    # Hot Region thresholds
    hot_lower_threshold = 8
    hot_upper_threshold = 12
    hot_mask = process_mask_hot(gt_original, hot_lower_threshold, hot_upper_threshold)

    # Cold Region thresholds
    cold_lower_threshold = 0.1
    cold_upper_threshold = 1
    cold_mask = process_mask_cold(gt_original, cold_lower_threshold, cold_upper_threshold,25,10)
    show_mask_overlay(ground_truth_image, hot_mask, cold_mask)

    if np.any(hot_mask):
        mean_activity_ground_truth_hot = np.mean(ground_truth_image[hot_mask==1])
        print(mean_activity_ground_truth_hot)
        mean_activity_target_hot = np.mean(target_image[hot_mask==1])
        print(mean_activity_target_hot)
        AR = mean_activity_target_hot / mean_activity_ground_truth_hot
    else:
        AR = None

    if np.any(cold_mask):
        mean_activity_ground_truth_cold = np.mean(ground_truth_image[cold_mask==1])
        mean_activity_target_cold = np.mean(target_image[cold_mask==1])
        RB = (mean_activity_target_cold - mean_activity_ground_truth_cold) / mean_activity_ground_truth_cold
    else:
        RB = None

    return AR*100, RB*100

def compute_image_roughness(ground_truth_image, target_image,gt_original):
    # Define thresholds for gray matter in the ground truth image
    gray_matter_lower_threshold = 1.1  # Adjust as needed
    gray_matter_upper_threshold = 4.5  # Adjust as needed

    # Create a binary mask for gray matter in the ground truth image
    gray_matter_mask = (gt_original > gray_matter_lower_threshold) & (gt_original < gray_matter_upper_threshold)

    # Find all connected components (regions) in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_matter_mask.astype(np.uint8), connectivity=8)

    # Filter out small regions (less than 6 pixels)
    large_regions_mask = np.zeros_like(gray_matter_mask, dtype=np.uint8)
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        if stats[i, cv2.CC_STAT_AREA] >= 10:
            large_regions_mask[labels == i] = 1

    # Erode the mask of large regions by 1 pixel to remove the border
    kernel = np.ones((4, 4), np.uint8)  # 3x3 kernel for erosion
    eroded_mask = cv2.erode(large_regions_mask, kernel, iterations=1)
    
    # Extract pixel values from the target image using the processed mask
    masked_target_values = target_image[eroded_mask > 0]
    mean_intensity_roi = np.mean(masked_target_values)
    variance_roi = np.var(masked_target_values, ddof=1)  # Corrected variance

    # Compute the Image Roughness (IR)
    L = len(masked_target_values)
    if L > 0:
        IR = np.sqrt(variance_roi) / mean_intensity_roi
        return IR*100
    else:
        return None  # In case no valid region exists


def compute_crc(ground_truth_image, target_image,gt_original):
    # Define thresholds for gray matter in the ground truth image
    gray_matter_lower_threshold = 5.0  # Adjust as needed
    gray_matter_upper_threshold = 8.5  # Adjust as needed

    # Create a binary mask for gray matter in the ground truth image
    gray_matter_mask = (gt_original > gray_matter_lower_threshold) & (gt_original < gray_matter_upper_threshold)

    # Find all connected components (regions) in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_matter_mask.astype(np.uint8), connectivity=8)

    # Filter out small regions (less than 6 pixels)
    large_regions_mask = np.zeros_like(gray_matter_mask, dtype=np.uint8)
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        if stats[i, cv2.CC_STAT_AREA] >= 2:
            large_regions_mask[labels == i] = 1

    # Erode the mask of large regions by 1 pixel to remove the border
    kernel = np.ones((1, 1), np.uint8)  # 3x3 kernel for erosion
    eroded_mask = cv2.erode(large_regions_mask, kernel, iterations=1)
    
  
    # Extract pixel values from the target image using the processed mask
    mean_target = np.mean(target_image[eroded_mask > 0])
    mean_real=np.mean(ground_truth_image[eroded_mask > 0])
    
    
    white_matter_lower_threshold = 1.1  # Adjust as needed
    white_matter_upper_threshold = 4.5  # Adjust as needed

    # Create a binary mask for gray matter in the ground truth image
    white_matter_mask = (gt_original > white_matter_lower_threshold) & (gt_original < white_matter_upper_threshold)

    # Find all connected components (regions) in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_matter_mask.astype(np.uint8), connectivity=8)

    # Filter out small regions (less than 6 pixels)
    large_regions_mask = np.zeros_like(white_matter_mask, dtype=np.uint8)
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        if stats[i, cv2.CC_STAT_AREA] >= 10:
            large_regions_mask[labels == i] = 1

    # Erode the mask of large regions by 1 pixel to remove the border
    kernel = np.ones((4, 4), np.uint8)  # 3x3 kernel for erosion
    eroded_mask = cv2.erode(large_regions_mask, kernel, iterations=1)
    
    # Extract pixel values from the target image using the processed mask
    mean_target_white = np.mean(target_image[eroded_mask > 0])
    mean_real_white=np.mean(ground_truth_image[eroded_mask > 0])
    
    crc=((mean_target/mean_target_white) -1)/((mean_real/mean_real_white) -1)
    
    return crc


def compute_std_white(ground_truth_image, target_image,gt_original):
    
    
    white_matter_lower_threshold = 1.1  # Adjust as needed
    white_matter_upper_threshold = 4.5  # Adjust as needed

    # Create a binary mask for gray matter in the ground truth image
    white_matter_mask = (gt_original > white_matter_lower_threshold) & (gt_original < white_matter_upper_threshold)

    # Find all connected components (regions) in the mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_matter_mask.astype(np.uint8), connectivity=8)

    # Filter out small regions (less than 6 pixels)
    large_regions_mask = np.zeros_like(white_matter_mask, dtype=np.uint8)
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        if stats[i, cv2.CC_STAT_AREA] >= 10:
            large_regions_mask[labels == i] = 1

    # Erode the mask of large regions by 1 pixel to remove the border
    kernel = np.ones((4, 4), np.uint8)  # 3x3 kernel for erosion
    eroded_mask = cv2.erode(large_regions_mask, kernel, iterations=1)
    
 
    std_white=np.std(target_image[eroded_mask > 0]-ground_truth_image[eroded_mask > 0])
    
    return std_white

def get_circular_statastic(img_it, img_gt, size=0.2):

    assert(size>0 and size<1)

    ftimage_it = np.fft.fft2(img_it)
    ftimage_it = abs(np.fft.fftshift(ftimage_it))

    ftimage_gt = np.fft.fft2(img_gt)
    ftimage_gt = abs(np.fft.fftshift(ftimage_gt))

    m_data = ftimage_it/(ftimage_gt+1e-8)
    m_data = np.clip(m_data, 0, 1)

    h,w = m_data.shape

    center = (int(w/2), int(h/2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    avg_mask_list = []
    pre_mask = np.zeros((h,w))
    for sz in np.linspace(size, 1, int(1/size)):

        radius = center[0]*sz#pow(center[0]**2+center[1]**2,0.5)
        mask = dist_from_center <= radius
        mask = mask.astype(np.int32)

        mask_sz = (mask-pre_mask).astype(np.int32)
        pre_mask = mask

        avg_mask_list.append(np.sum(mask_sz*m_data)/np.sum(mask_sz))

    return avg_mask_list
def reconstruction_osem(beta,nb_img):
    os.environ['CASTOR_CONFIG'] = "/home/user/Documents/phd/codes_and_data/castor_gitlab/config"
    castor_command = ["/home/user/Documents/phd/codes_and_data/castor_gitlab/bin/castor-recon", "-df", "/home/user/Documents/phd/codes_and_data/generalization_data/web3M/data0_1/data0_1.cdh",
                    "-dim", "160,160,1", "-vox", "2.,2.,2.",
                     "-it", "7:28", "-conv", "gaussian,4,1,3.5::psf",
                    "-dout", "/home/user/Documents/phd/codes_and_data/brain_phantom/output/out_brain_cropp"]

    # Execute the CASTOR command as a subprocess
    subprocess.run(castor_command, stdout=subprocess.DEVNULL)
    root='/home/user/Documents/phd/codes_and_data/brain_phantom/output/out_brain_cropp/out_brain_cropp_it2.img'
    fo = open(root, "rb")
    img = np.fromfile(fo, dtype=np.float32).reshape((160,160))
    return img

img=reconstruction_osem(0.5,12)
plt.imshow(img,cmap='gray_r',vmin=0,vmax=200)
plt.colorbar()
plt.show()
recon_osem_image=img

"""
This script processes hyperspectral image data from a specified input directory, extracts relevant information, 
and saves the results in an output directory. The script performs the following steps:
1. Iterates through folders in the input directory.
2. For each folder, it identifies and processes .hdr and .dat files.
3. Reads the hyperspectral image data and extracts specific bands to create an RGB image.
4. Normalizes and saves the RGB image as a PNG file.
5. Extracts HMI (Hyperspectral Microscopy Imaging) data, including mask, image intensity, wavelengths, 
    single cell data, and mean spectra.
6. Saves the extracted mask and spectra data as .npy files.
Functions:
     process_folder(input_path, output_dir, label):
          Processes the folders in the input directory, extracts hyperspectral image data, 
          and saves the results in the output directory.
Parameters:
     input_path (str): Path to the input directory containing hyperspectral image data.
     output_dir (str): Path to the output directory where results will be saved.
     label (str): Label for the output data.
Usage:
     Specify the path to your data directory in the `input_dir` variable and run the script.
"""

import os
import spectral.io.envi as envi
from PIL import Image
import numpy as np
import bacterial_hmi as bhmi
from bacterial_hmi.types import SegmentationResults
from bacterial_hmi.utils import read_spectra_data, normalize_spectra_data
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import os


COPY_DIR = '/mnt/projects/bhatta70/VBNC-Detection/outputs/Salmonella_serovars_rgb_full/'
redband = 112
greenband = 77
blueband = 27

def process_one_sample(hdr_path, dat_path, output_dir, classname):
    label = hdr_path.split('/')[-1].replace('.hdr', '')

    # check if files already exist
    if os.path.exists(os.path.join(output_dir, label + '.png')) and \
        os.path.exists(os.path.join(output_dir, label + '_mask.npy')) and \
        os.path.exists(os.path.join(output_dir, label + '_spectra_mean.npy')) and \
        os.path.exists(os.path.join(output_dir, label + '_single_cell_spectra.npy')):
        return
    img = envi.open(hdr_path, dat_path)
    # Adjust band indices based on your specific dataset
    band_red = img.read_band(redband)
    band_green = img.read_band(greenband)
    band_blue = img.read_band(blueband)
    # Stack the bands to form an RGB image
    rgb_image = np.dstack((band_red, band_green, band_blue))
    # Create and save the image
    rgb_image_normalized = ((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255).astype('uint8')
    img_png = Image.fromarray(rgb_image_normalized.astype('uint8'), 'RGB')
    img_png.save(os.path.join(output_dir, label + '.png'))
    print("Saved image:", os.path.join(output_dir, label + '.png'))

    ##### Extract HMI data #####

    if os.path.exists(os.path.join(COPY_DIR, classname,label + '_mask.npy'))\
             and os.path.exists(os.path.join(COPY_DIR, classname,label + '_spectra_mean.npy'))\
             and os.path.exists(os.path.join(COPY_DIR, classname,label + '_single_cell_spectra.npy')):
        # copy the mask and spectra to the output directory
        cmd = f'cp {os.path.join(COPY_DIR, classname,label + "_mask.npy")} {output_dir}'
        os.system(cmd)
        cmd = f'cp {os.path.join(COPY_DIR, classname,label + "_spectra_mean.npy")} {output_dir}'
        os.system(cmd)
        cmd = f'cp {os.path.join(COPY_DIR, classname,label + "_single_cell_spectra.npy")} {output_dir}'
        os.system(cmd)
        return
        
    result = bhmi.extract_hmi_data(dat_path)
    mask = result.mask
    # img_intensity = result.image_intensity
    # wavelengths = result.wavelengths

    single_cell_data = result.single_cell_data
    single_cell_spectra = np.array([celldata.mean_spectra for celldata in single_cell_data])
    mean_spectra = np.mean(single_cell_spectra, axis=0)

    # Save the mask and spectra
    np.save(os.path.join(output_dir, label + '_mask.npy'), mask)
    spectra_mean_path = os.path.join(output_dir, label + '_spectra_mean.npy')
    print("spectra_mean_path:", spectra_mean_path)
    np.save(spectra_mean_path, mean_spectra)
    np.save(os.path.join(output_dir, label + '_single_cell_spectra.npy'), single_cell_spectra)



def process_folder(input_path, output_dir, label):
    counter = 1
    for folder in os.listdir(input_path):
        parent_folder_path = os.path.join(input_path, folder)
        print("Processing parent_folder_path:", parent_folder_path)
        if not os.path.isdir(parent_folder_path) or 'extra' in folder:
            continue
        classname = folder
        print("Processing classname:", classname)
        # make new folder for the class
        output_path = os.path.join(output_dir, classname)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        sample_label = f'{classname}'
        for inner_folder in os.listdir(parent_folder_path):
            folder_path = os.path.join(parent_folder_path, inner_folder)
            print("Processing folder_path:", folder_path)
            if os.path.isdir(folder_path):
                hdr_path = None
                dat_path = None
                # Find the .hdr and .dat files in the folder
                for file in os.listdir(folder_path):
                    if file.endswith('.hdr'):
                        hdr_path = os.path.join(folder_path, file)
                    elif file.endswith('.dat'):
                        dat_path = os.path.join(folder_path, file)
                # Process the files if both are found
                if hdr_path and dat_path:
                    process_one_sample(hdr_path, dat_path, output_path, classname)
                    counter += 1
 
# Specify the path to your data directory here


# process_folder(input_dir, output_dir, output_label)
if __name__ == '__main__':
    
    input_dir = '/mnt/data/Salmonella_serovars'
    output_label = input_dir.split('/')[-1]  + f"_rgb_full_normalized"
    output_dir = f'./outputs/{output_label}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_folder(input_dir, output_dir, output_label)
    
    # process_one_sample('/mnt/projects/cifs/HMI/Salmonella_serovars/Kentucky/C1-S3-03-1009-SKentucky-300ms-16db/C1-S3-03-1009-SKentucky-300ms-16db.hdr',
    #                 '/mnt/projects/cifs/HMI/Salmonella_serovars/Kentucky/C1-S3-03-1009-SKentucky-300ms-16db/C1-S3-03-1009-SKentucky-300ms-16db.dat',
    #                 '/mnt/projects/bhatta70/VBNC-Detection/outputs/Salmonella_serovars_all/Kentucky/')

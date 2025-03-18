from .image_processing import *
from .types import SegmentationResults
from .utils import read_spectra_data, normalize_spectra_data
import time

best_model = None
def subfolder_path_package(packagename, subfoldername):
    import importlib.util
    # Get the spec object for the package
    spec = importlib.util.find_spec(packagename)
    # Get the path of the package
    package_path, _, _ = fileparts(spec.origin)
    # Get the path of the subfolder
    subfolder_path = os.path.join(package_path, subfoldername)
    # Print the path of the subfolder
    return subfolder_path

def get_model():
    model_dir = subfolder_path_package("bacterial_hmi", "models")
    best_model = load_AR2UNet_model(model_dir)
    return best_model, model_dir


def extract_hmi_data(envi_file, min_gof=8.0, outdir=r"./qsaru_output", suboutdir_prefix="results"):
    """Segment & classify bacteria of four species (E. coli, L. innocua, S. aureus, S. typhimurium) from a datacube and display the results.

    Parameters
    ----------
    filename : filename of datacube to classify.
    min_gof: min of allowed GOF for single-cell selection
	outdir: root output folder path
    suboutdir_prefix: prefix of output folder name

    """
    global best_model
    model_dir = subfolder_path_package("bacterial_hmi", "models")

    sigma = 15 
    # 1 Preparing segmentation
    # 	• Get intensity image and its processed image for segmentation
    img_intensity, img_input = get_intensity_image(envi_file, sigma=sigma)
    print("1. Got the intensity image and deblurred it")

    # 2 Segmenting cell candidates with DL
    if best_model is None:
        best_model = load_AR2UNet_model_ptch(model_dir)
    # 	• Produce mask image by running AGR2U-Net
    mask, predicted = predict_bacteria_cell_mask_ptch(img_input, best_model)


    # 3 Finding single cells
    org_img_dim = img_intensity.shape[:2] 
    mask = find_single_cells(mask, min_gof=min_gof, org_img_dim=org_img_dim)


    single_cell_results = get_single_cell_spectra(envi_file, mask)
    raw_mean_spectra = np.array([celldata.mean_spectra for celldata in single_cell_results])
    normalized_mean_spectra = normalize_spectra_data(raw_mean_spectra)
    for i in range(len(single_cell_results)):
        single_cell_results[i].mean_spectra = normalized_mean_spectra[i]

    wavelengths = get_wavelengths(envi_file)

    envi_file = os.path.abspath(envi_file)
    # single_cell_spectra = read_spectra_data(out_datafile)
    # single_cell_spectra = normalize_spectra_data(single_cell_spectra)
    segmentation_result = SegmentationResults(
        dat_filename=envi_file,
        single_cell_data=single_cell_results,
        mask=mask,
        image_intensity=img_intensity,
        wavelengths=wavelengths
    )
    return segmentation_result

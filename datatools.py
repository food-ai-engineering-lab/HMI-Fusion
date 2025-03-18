import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tifffile as tiff
from PIL import Image
import skimage.exposure as exposure
from glob import glob
import pytorch_lightning as pl
from segment_save import process_one_sample


# Define allowed image extensions.
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# Input image dimensions (match this with your dataset)
max_px = 1000
min_px = 1000

# Define a class for data augmentation
class Transforms:
    # Training set transforms include several augmentation techniques
    def __init__(self, train=True):
        if train:
            self.tf = A.Compose([
                A.ToFloat(max_value=255),
                A.Resize(min_px, min_px),
                A.ToGray(p=1.0),

                ## Model v1
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.5,
                                           brightness_by_max=True,
                                           p=0.7),
                
                ## Model v2 only
                #A.Transpose(p=0.5),
                #A.Blur(blur_limit=7, p=0.5),
                #A.MedianBlur(blur_limit=7, p=0.5),
                #A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
                #A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
                                #border_mode=4, value=None, mask_value=None, 
                                #normalized=False, p=0.5),
                #A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
                                    #interpolation=1, border_mode=4, value=None, 
                                    #mask_value=None, p=0.5),
                #A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
                         #shear=None, interpolation=1, mask_interpolation=0, cval=0, 
                         #cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
                #A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
                              #mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
                A.Sharpen(alpha=(0.2, 0.8), lightness=(0.5, 1.0), p=0.5),

                #A.Defocus(radius=(1, 1.2), alias_blur=(0.005, 0.01), p=0.1),
                #A.ZoomBlur(max_factor=1.02, step_factor=(0.001, 0.002), p=0.1),
                #A.GaussNoise(var_limit=(0.1, 2.0), mean=0, per_channel=True, p=0.1),
                #A.ElasticTransform(alpha=0.05, sigma=0.005, alpha_affine=0.005, p=0.1),
                A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=2, p=0.1),
                #A.CoarseDropout(max_holes=1, max_height=1, max_width=1, min_holes=None, min_height=None, min_width=None, fill_value=0, p=0.1),


                ## Model v2a only
                # A.Transpose(p=0.6),
                # A.Blur(blur_limit=3, p=0.4),
                # # A.MedianBlur(blur_limit=3, p=0.4),
                # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.4),
                # A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
                #                 border_mode=4, value=None, mask_value=None, 
                #                 normalized=False, p=0.5),
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
                #                     interpolation=1, border_mode=4, value=None, 
                #                     mask_value=None, p=0.4),
                # A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
                #          shear=None, interpolation=1, mask_interpolation=0, cval=0, 
                #          cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
                # A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
                #               mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
                # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    
                ## Model v3 only
                # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                # A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.5),
                # A.ZoomBlur(max_factor=1.31, step_factor=(0.01, 0.03), p=0.5),
                # A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                #                num_shadows_upper=2, shadow_dimension=5, p=0.5),
                # A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
                # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
                # A.RandomScale(scale_limit=(-0.7,0), interpolation=1, p=0.5), #scaling factor range=(0.3,1) ~obj lens
                # A.PadIfNeeded(min_height=max_px, min_width=max_px, border_mode=0, value=(0,0,0)),

                ToTensorV2()]) # numpy HWC image -> pytorch CHW tensor 
        # Validation set transforms only include basic conversions and resizing
        else:
            self.tf = A.Compose([
                A.ToFloat(max_value=255),
                A.Resize(min_px, min_px),
                A.ToGray(p=1.0),
                ToTensorV2()])

    # Allow the class instance to be called as a function to transform images
    def __call__(self, img, *args, **kwargs):
        return self.tf(image=np.array(img))['image']


# Function to convert single-channel tif images to RGB
def tif1c_to_tif3c(path):
    """Converts single-channel tif images to RGB

    Args:
        path (string): A root folder containing original input images

    Returns:
        img_tif3c (numpy.ndarray): tif image converted to RGB
    """
    img_tif1c = tiff.imread(path)
    img_tif1c = np.array(img_tif1c)
    img_rgb = np.zeros((img_tif1c.shape[0],img_tif1c.shape[1],3),dtype=np.uint8) # blank array
    img_rgb[:,:,0] = img_tif1c # copy img 3 times to make the format of img.shape=[H, W, C]
    img_rgb[:,:,1] = img_tif1c
    img_rgb[:,:,2] = img_tif1c
    img_tif3c = img_rgb
    
    # normalize image to 8-bit range
    img_norm = exposure.rescale_intensity(img_rgb, in_range='image', out_range=(0,255)).astype(np.uint8)
    img_tif3c = img_norm
    
    print(img_tif3c.dtype)
    return img_tif3c

def spectra_hmi_loader(path):
    """
    Custom loader for a single sample. Given an image path, it loads:
      - the image (as a PIL Image),
      - a mask (from a .npy file with '_mask.npy' suffix), and
      - a spectra_mean (from a .npy file with '_spectra_mean.npy' suffix).
    """
    # Load the image and ensure it's in RGB format.
    image = Image.open(path).convert("RGB")
    
    # Derive the mask and spectra_mean file paths.
    mask_path = path.replace('.png', '_mask.npy')
    spectra_mean_path = path.replace('.png', '_spectra_mean.npy')
    
    if not os.path.exists(mask_path) or not os.path.exists(spectra_mean_path):
        # raise FileNotFoundError(f"Mask or spectra_mean file not found for image: {path}")
        sample_fname = mask_path.split('/')[-1].replace('_mask.npy', '')
        classname = os.path.basename(os.path.dirname(path))
        dataset_path = '/mnt/projects/cifs/HMI/Salmonella_serovars/'
        hdr_path = os.path.join(dataset_path, classname, sample_fname, f'{sample_fname}.hdr')
        dat_path = os.path.join(dataset_path, classname, sample_fname, f'{sample_fname}.dat')
        save_dir = os.path.join('/mnt/projects/bhatta70/VBNC-Detection/outputs/Salmonella_serovars_all', classname)
        print(f"Processing HDR_path: {hdr_path}, DAT_path: {dat_path}, Save_dir: {save_dir}")
        process_one_sample(hdr_path, dat_path, save_dir)
    
    mask = np.load(mask_path).astype(np.uint8)
    spectra_mean = np.load(spectra_mean_path).astype(np.float32)
    # minmax 
    spectra_mean = (spectra_mean - np.min(spectra_mean)) / (np.max(spectra_mean) - np.min(spectra_mean))
    
    return image, mask, spectra_mean


class SpectraHMIDataset(datasets.DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=spectra_hmi_loader, extensions=IMG_EXTENSIONS,
                 is_valid_file=None):
        """
        Args:
            root (string): Root directory path. It should have one subdirectory per class.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. This will only be applied to the image.
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            loader (callable): Function to load a sample given its path.
            extensions (tuple[string]): A tuple of allowed extensions (only files with these
                extensions will be loaded).
            is_valid_file (callable, optional): A function that takes path of an Image file
                and check if the file is a valid file (used instead of extensions).
        """
        super(SpectraHMIDataset, self).__init__(
            root, loader, extensions, transform=transform,
            target_transform=target_transform, is_valid_file=is_valid_file
        )
        # For compatibility with ImageFolder.
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Overrides the base class __getitem__ to separately transform the image.
        The loader returns a tuple (image, mask, spectra_mean), and the target is the class index.
        """
        path, target = self.samples[index]
        image, mask, spectra_mean = self.loader(path)
        
        # Apply the transform only to the image.
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Return a tuple: ( (image, mask, spectra_mean), class_index )
        return (image, mask, spectra_mean), target

# Define a PyTorch Lightning data module for handling dataset
class DataModule(pl.LightningDataModule):
    def __init__(self, root: str, dl_workers: int = 0, batch_size=4, sampler: str = None):
        super().__init__()
        
        self.transforms = Transforms(train=True)
        # self.train_transforms = Transforms(train=True)
        # self.val_transforms = Transforms(train=False)
        self.root = root
        self.workers = dl_workers
        self.batch_size = batch_size
        
        # Load sampler if it exists
        if sampler is not None:
            self.sampler = torch.load(sampler)
            self.sampler.generator = None
        else:
            self.sampler = None
            
    # Setup data for training/validation/testing
    def setup(self, stage: str = None):
        raise NotImplementedError("DataModule setup method must be implemented in subclass.")
        # if stage == "fit" or stage is None:
        #     ds = datasets.ImageFolder(self.root, transform=self.transforms)
        #     # ds = datasets.ImageFolder(self.root, transform=self.train_transforms)
        #     # ds = datasets.ImageFolder(self.root, loader=tif1c_to_tif3c) # for imgs.tif
        # if stage == "test" or stage is None:
        #     ds = datasets.ImageFolder(self.root, transform=self.transforms)
        #     # ds = datasets.ImageFolder(self.root, transform=self.val_transforms)
            
        # # Create train and validation splits
        # train_size = int(np.floor(len(ds)*0.7))
        # val_size = int(len(ds)-int(np.floor(len(ds)*0.7)))
        # self.train, self.val = random_split(ds, [train_size, val_size], torch.Generator().manual_seed(111821))

    # Define methods to retrieve data loaders for each dataset
    def train_dataloader(self):
        if self.sampler is None:
            return DataLoader(self.train, batch_size=self.batch_size, 
                              shuffle=True, num_workers=self.workers, pin_memory=False,
                              drop_last=True)
        else:
            return DataLoader(self.train, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.workers, pin_memory=False,
                              drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False,
                          drop_last=True)

class SpectralHMIDataModule(DataModule):
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            ds = SpectraHMIDataset(self.root, transform=self.transforms)
            
            # Create train and validation splits
            train_size = int(np.floor(len(ds)*0.7))
            val_size = int(len(ds)-int(np.floor(len(ds)*0.7)))
            self.train, self.val = random_split(ds, [train_size, val_size], torch.Generator().manual_seed(111821))
        elif stage == "test":
            ds = SpectraHMIDataset(self.root, transform=self.transforms)
            self.test = ds

class McolonyDataModule(DataModule):
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            ds = datasets.ImageFolder(self.root, transform=self.transforms)
            # ds = datasets.ImageFolder(self.root, transform=self.train_transforms)
            # ds = datasets.ImageFolder(self.root, loader=tif1c_to_tif3c) # for imgs.tif
        if stage == "test" or stage is None:
            ds = datasets.ImageFolder(self.root, transform=self.transforms)
            # ds = datasets.ImageFolder(self.root, transform=self.val_transforms)
            
        # Create train and validation splits
        train_size = int(np.floor(len(ds)*0.7))
        val_size = int(len(ds)-int(np.floor(len(ds)*0.7)))
        self.train, self.val = random_split(ds, [train_size, val_size], torch.Generator().manual_seed(111821))
# Define a class for handling test data
class McolonyTestData(object):
        def __init__(self, root):
            self.root = root
            self.tform = A.Compose([A.ToFloat(max_value=255), A.Resize(min_px, min_px), A.ToGray(p=1.0), ToTensorV2()])
            file_list = glob(root+'*.png')
            file_list.sort()
            self.img_idx = [os.path.basename(x) for x in file_list]

        def __getitem__(self, idx):            
            ## load images and labels
            fname = self.img_idx[idx]
            im = Image.open(self.root+fname)
            # im = im.convert("RGB")
            im = self.tform(image=np.array(im))
            return im, fname

        def __len__(self):
            return len(self.img_idx)
        
class SpectralHMITestData:
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Root directory containing test images in subdirectories.
            transform (callable, optional): Transform to apply to the image. If None,
                a default validation transform is applied.
        """
        self.root = root
        # Use provided transform or default to the validation transforms in Transforms
        self.transform = transform if transform is not None else Transforms(train=False)
        # Recursively find all PNG files under root
        self.file_list = glob(os.path.join(root, '*.png'), recursive=True)
        self.file_list.sort()
        # Save just the base filenames (for reference)
        self.img_idx = [os.path.basename(x) for x in self.file_list]

    def __getitem__(self, idx):
        # Get the full path of the image sample
        path = self.file_list[idx]
        # Load the sample using the custom loader, which returns (image, mask, spectra_mean)
        image, mask, spectra_mean = spectra_hmi_loader(path)
        
        # Apply transformation to the image (if provided)
        if self.transform:
            image = self.transform(image)
        
        # Return a tuple: ( (image, mask, spectra_mean), filename )
        return (image, mask, spectra_mean), self.img_idx[idx]

    def __len__(self):
        return len(self.file_list)

from pydantic import BaseModel as _BaseModel
from typing import List, Tuple
import numpy as np

class BaseModel(_BaseModel):
    # Parent BaseModel that allows arbitrary types
    class Config:
        arbitrary_types_allowed = True

class SingleCellData(BaseModel):
    contour: np.ndarray
    mean_spectra: np.ndarray


class SegmentationResults(BaseModel):
    dat_filename: str
    single_cell_data: List[SingleCellData]
    mask: np.ndarray
    image_intensity: np.ndarray
    wavelengths: List[float]

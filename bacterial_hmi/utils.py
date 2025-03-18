import os
import numpy as np
import pandas as pd
# import keras
# from keras import backend as K


def dice_coef(y_true, y_pred):
    """Computes Dice coefficient for a Keras loss
    """
    # The “+1” term has two effects: (1) shift the range from
    # [0,1] to [0,0.5], (2) prevent loss DL(p,^p)=0, when p=0 and ^p>0.
    # The disadvantage is when p=0, we get 1−1^p+1=^p^p+1.
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    mask_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    ret = (2. * intersection + smooth) / (mask_sum + smooth)
    return ret


def mean_iou(y_true, y_pred):
    """Computes mean IOU for a Keras loss
    """
    # The “+1” term has two effects: (1) shift the range from
    # [0,1] to [0,0.5], (2) prevent loss DL(p,^p)=0, when p=0 and ^p>0.
    # The disadvantage is when p=0, we get 1−1^p+1=^p^p+1.
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    mask_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    union = mask_sum - intersection
    ret = (intersection + smooth) / (union + smooth)
    return ret


def dice_loss(y_true, y_pred):
    """Computes Dice loss for a Keras loss
    """
    pp = dice_coef(y_true, y_pred)
    return -K.log(pp)


def mean_iou_loss(y_true, y_pred):
    """Computes Mean IOU loss for a Keras loss.
    -Log was used for better convergence
    """
    pp = mean_iou(y_true, y_pred)
    return -K.log(pp)


def fileparts(filename):
    """Separate file path to folder, file name, and extension.
     Args:
          filename: file name string
     Output:
          folder, filename, file_extension
    """
    path, file_extension = os.path.splitext(filename)
    folder, filename = os.path.split(path)
    return folder, filename, file_extension


def msc(x, xref=None):
    """Multiplicative Scatter Correction converted from
            Matlab code by Cleiton A. Nunes

    x_msc = msc(x,xref)

    input
    x (samples x variables)      spectra to correct
    xref (1 x variables)         reference spectra (in general mean(x) is used)

    Output
    x_msc (samples x variables)  corrected spectra
    """
    m, n = x.shape

    if xref is None:
        rs = x.mean(axis=0)
    else:
        rs = xref

    cw = np.ones((1, n))
    mz = np.hstack((np.ones((n,1)), np.reshape(rs, (n, 1))))
    mm, nm = mz.shape
    wmz = mz * (np.transpose(cw) @ np.ones((1, nm)))  # '*' elementwise multiply and '@' matrix multiply
    wz = x * (np.ones((m, 1)) @ cw)
    z = np.transpose(wmz) @ wmz
    z = z.astype(np.float)
    u, s, v = np.linalg.svd(z)
    sd = s  # instead of np.transpose(np.diag(s))
    cn = pow(10, 12)
    ms = sd[0] / math.sqrt(cn)
    cs = [max(sdi, ms) for sdi in sd]
    cz = u @ (np.diag(cs)) @ np.transpose(v)
    zi = np.linalg.inv(cz)
    b = zi @ np.transpose(wmz) @ np.transpose(wz)
    B = np.transpose(b)
    x_msc = x
    p = np.reshape(B[:, 0], (B.shape[0],1))
    x_msc = x_msc - (p @ np.ones((1, mm)))
    p = np.reshape(B[:, 1], (B.shape[0],1))
    x_msc = x_msc / (p @ np.transpose(np.ones((mm, 1))))
    return x_msc


def snv(input_data):
    """Define a new array and populate it with the corrected data by Daniel Pelliccia"""
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :], ddof=1)
    return data_snv

def snv_row(input_data):
    """Define a new array and populate it with the corrected data by Daniel Pelliccia"""
    data_snv = (input_data - np.mean(input_data)) / np.std(input_data, ddof=1)
    return data_snv

def difference(data, gap, direction):
    """
    DIFFERENCE implements the first order derivative.

        out = Difference( data, gap, direction)

        The first "gap" points are discard.
        Thus, the size of out is (n-gap).
    """
    if len(data.shape) != 2 or gap <= 0:
        raise ValueError('not supported dimension. should be 2')
    out = None
    if min(data.shape) == 1:
            out = data[gap:] - data[:-gap]
    else:
        if direction == 'column':
            out = data[gap:, :] - data[:-gap, :]
        elif direction == 'row':
            out = data[:, gap:] - data[:, :-gap]
        else:
            raise ValueError('not supported direction')
    return out


def detrend_polyfit(X, order=2):
    """
    DETRENDPOLYFIT Removes a non-linear trend from each column of a matrix. The nonlinearity is
       estimated with polynomial fit.

        Y = detrendpolyfit(X, *varargin)

        X is the input matrix.
        varargin is an integer number representing polynomial order. Its default value is 2.

        Returns values subtracted polynomial fit of order varargin from the input data X
    """
    if len(X.shape) < 2:
        X = np.reshape(X, (1, len(X)))
    Y = np.zeros(X.shape)
    for i in range(X.shape[1]):
        x = X[:, i]
        rng = [z+1 for z in range(len(x))]
        p = np.polyfit(np.transpose(rng), x[:], order)
        Y[:, i] = x[:]-np.polyval(p, np.transpose(rng))
    return Y


def deriv_norrisgap(X, gap, direction):
    """
    first derivative using Norris Gap
    :param X: m x n matrix
    :param gap: gap < n
    :param direction: 'row' or 'column'
    :return: Xout
    """
    if len(X.shape) != 2:
        raise ValueError('not supported dimension. should be 2')

    gap = gap + 1
    if min(X.shape) == 1:
        out = X[gap:] - X[:-gap]
    else:
        if direction == 'column':
            out = X[gap:, :] - X[:-gap, :]
        elif direction == 'row':
            out = X[:, gap:] - X[:, :-gap]

    out = out / gap
    Xout = rearrange_norrisgap(X, out)
    return Xout


def rearrange_norrisgap(X, out):
    """Re-arrage norris gap output"""
    M1, N1 = X.shape
    M2, N2 = out.shape
    if M2 < M1:
        # vertical direction
        offset = M1 - M2
        numPts = M2
        direction = 0
    elif N2 < N1:
        # horizontal direction
        offset = N1 - N2
        numPts = N2
        direction = 1
    else:
        raise ValueError('Error in Derivative_NorisGap')
    pos1 = np.int32(offset / 2)
    pos2 = np.int32(pos1 + numPts)
    Xout = np.float64(np.zeros_like(X))
    if direction == 0:
        Xout[pos1:pos2, :] = out
    elif direction == 1:
        Xout[:, pos1:pos2] = out
    return Xout


def movingAverage(x, gap, dir, *varargin):
    """
    MOVINGAVERAGE computes mnoving average for smoothing a vector or each line in a matrix.

    y = movingAverage(x, gap, dir, varargin)

      X is the input vector or matrix.
      GAP is the window size (odd number), i.e. the number of points to take an average.
      dir is a direction. For example, 'col' means columnwise smoothing and
          'row' means rowwise smoothing.
      varargin can be 'same' or 'valid'. Use 'same' to produce same output dimension as the one of input x.
          default value is 'valid' to generate n - gap + 1 output elements.

      Returns y with n - gap + 1 elements or n elements according to varargin. With varargin='valid',
          both boundary points ((gap-1)/2 each ) are set to
          zeros (in fact, discard). gap should be an odd number.


    if gap % 2 == 0:
        raise ValueError("use an odd number for the moving average's point gap")

    n = gap // 2

    if len(varargin) > 0 and varargin[0] == 'same':
        if x.ndim == 1:
            if x.shape[0] < x.shape[1]:
                y = colfilt(x, [1, gap], 'sliding', @ mean)
            else:
                y = colfilt(x, [gap, 1], 'sliding', @ mean)
            for i=1:n:
                y(i) = mean(x(1:i + n))
            ny = length(y)
            for i=1:n:
                y(ny - i + 1) = mean(x((ny - i - n + 1):ny))
        elif ismatrix(x):
            if isequal(dir, 'row'):
                y = colfilt(x, [1, gap], 'sliding', @ mean)
                for i=1:n:
                    y(:, i)      = mean(x(:, 1: i + n), 2)
                ny = size(y, 2)
                for i=1:n:
                    y(:, ny - i + 1) = mean(x(:, (ny - i - n + 1): ny), 2)
            elif isequal(dir, 'col'):
                y = colfilt(x, [gap, 1], 'sliding', @ mean)
                for i=1:n:
                    y(i,:)      = mean(x(1: i + n,:), 1)
                ny = size(y, 1)
                for i=1:n:
                    y(ny - i + 1,:) = mean(x((ny - i - n + 1): ny,:), 1)
        else:
            raise ValueError('SHAPE method is not correct. Use valid or same.')
        end

    elif isempty(varargin) or strcmp(varargin{1}, 'valid'):
        if isvector(x):
            if isrow(x):
                y = colfilt(x, [1, gap], 'sliding', @ mean)
            else:
                y = colfilt(x, [gap, 1], 'sliding', @ mean)
            y = y(n + 1:end - n)
        elif ismatrix(x):
            if isequal(dir, 'row'):
                y = colfilt(x, [1, gap], 'sliding', @ mean)
                y = y(:, n + 1: end - n)
            elif isequal(dir, 'col'):
                y = colfilt(x, [gap, 1], 'sliding', @ mean)
                y = y(n + 1:end - n,:)
        else:
            raise ValueError('x dimension should be at most 2D.')
    else:
        raise ValueError('SHAPE method is not correct. Use valid or same.')
    return y
"""
# data = np.float64(magic(5))
# res = msc(data, data.mean(axis=0))
# print(res)
# snv(data)
# ts = range(21)
# x = np.float64([3*math.sin(t) + t for t in ts])
# zz = deriv_norrisgap(magic(5), 1, 'row')
# print(zz)
# scipy.ndimage.generic_filter()


def expandarr(x,k):
    #make it work for 2D or nD with axis
    kadd = k
    if np.ndim(x) == 2:
        kadd = (kadd, np.shape(x)[1])
    return np.r_[np.ones(kadd)*x[0],x,np.ones(kadd)*x[-1]]


def movmoment(x, k, windowsize=3, lag='lagged'):
    """non-central moment
    Parameters
    ----------
    x : array
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : array
        k-th moving non-central moment, with same shape as x
    Notes
    -----
    If data x is 2d, then moving moment is calculated for each
    column.
    """

    from scipy import signal
    windsize = windowsize
    #if windsize is even should it raise ValueError
    if lag == 'lagged':
        #lead = -0 + windsize #windsize//2
        lead = -0# + (windsize-1) + windsize//2
        sl = slice((windsize-1) or None, -2*(windsize-1) or None)
    elif lag == 'centered':
        lead = -windsize//2  #0#-1 #+ #(windsize-1)
        sl = slice((windsize-1)+windsize//2 or None, -(windsize-1)-windsize//2 or None)
    elif lag == 'leading':
        #lead = -windsize +1#+1 #+ (windsize-1)#//2 +1
        lead = -windsize +2 #-windsize//2 +1
        sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    else:
        raise ValueError

    avgkern = (np.ones(windowsize)/float(windowsize))
    xext = expandarr(x, windsize-1)
    # Note: expandarr increases the array size by 2*(windsize-1)
    # sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    # print(sl)

    if xext.ndim == 1:
        return np.correlate(xext**k, avgkern, 'full')[sl]
        # return np.correlate(xext**k, avgkern, 'same')[windsize-lead:-(windsize+lead)]
    else:
        # print(xext.shape)
        # print(avgkern[:,None].shape)

        # try first with 2d along columns, possibly ndim with axis
        return signal.correlate(xext**k, avgkern[:,None], 'full')[sl, :]


def movmean(x, windowsize=3, mode="same"):
    """moving window mean
    Parameters
    ----------
    x : array
       time series data
    windsize : int
       window size
    mode : string
        "same" - output array has same size as input
        "valid" - output array has reduced ize from input containing non-truncated values
    Returns
    -------
    mk : array
        moving mean, with same shape as x
    Notes
    -----
    When mode="same"
    The window size is automatically truncated at the endpoints
    when there are not enough elements to fill the window.
    When the window is truncated, the average is taken over
    only the elements that fill the window.
    """
    if type(x) is not np.ndarray:
        raise ValueError("input should be numpy ndarray!")

    lag = 'centered'
    ret = movmoment(x, 1, windowsize=windowsize, lag=lag)

    # to handle truncated window operations correctly
    n = windowsize//2
    if x.ndim == 2 and min(x.shape) > 1:
        for i in range(n):
            ret[i, :] = np.mean(x[:(i + n + 1), :], axis=0)
            ret[-(i + 1), :] = np.mean(x[-(i + n + 1):, :], axis=0)
    elif x.ndim == 1 or (x.ndim == 2 and min(x.shape) == 1):
        for i in range(n):
            ret[i] = np.mean(x[:(i + n + 1)])
            ret[-(i + 1)] = np.mean(x[-(i + n + 1):])
        pass
    else:
        raise ValueError("input dimension should be <= 2!")

    # remove values calculated with truncated window size in "valid" mode
    if mode == "valid":
        if x.ndim == 2 and min(x.shape) > 1:
            ret = ret[n:-n,:]
        elif x.ndim == 1 or (x.ndim == 2 and min(x.shape) == 1):
            ret = ret[n:-n]
    return ret


def msc_diff_ma(x, gap=7):
    """MSC+Difference+Moving average"""
    res = msc(x)
    res = difference(res, gap, 'row')
    res = movmean(res.T, gap, 'same').T
    return res


def msc_diff(x, gap=7):
    """MSC+Difference+Moving average"""
    res = msc(x)
    res = difference(res, gap, 'row')
    return res


def msc_ma(x, gap=7):
    """MSC+Moving average"""
    res = msc(x)
    res = movmean(res.T, gap, 'same').T
    return res


def snvd_diff_ma(x, gap=7):
    """SNV+Detrending+Difference+Moving average"""
    res = snv(x)
    res = signal.detrend(res)
    res = difference(res, gap, 'row')
    res = movmean(res.T, gap, 'same').T
    return res


def snvd_diff(x, gap=7):
    """SNV+Detrending+Difference+Moving average"""
    from statsmodels.tsa.tsatools import detrend
    res = snv(x).astype(np.float64)
    res1 = detrend(res, axis=1)
    res2 = difference(res1, gap, 'row')
    return res2


def snvd_ma(x, gap=7):
    """SNV+Detrending++Moving average"""
    res = snv(x)
    res = signal.detrend(res)
    res = movmean(res.T, gap, 'same').T
    return res


def load_AR2UNet_model(model_dir):
    """load trained AR2U-Net model from model_dir folder"""
    import tensorflow as tf
    # Predict mask with DL model
    with tf.device('/cpu:0'):
        best_model = keras.models.load_model(fr"{model_dir}/segment_model",
										 custom_objects={
											 'mean_iou_loss': mean_iou_loss,
											 'mean_iou': mean_iou,
											 'dice_coef': dice_coef,
										 })
    return best_model


def load_AR2UNet_model_ptch(model_dir):
    """load trained AR2U-Net model from model_dir folder"""
    import onnx
    from onnx2pytorch import ConvertModel

    best_model = ConvertModel(onnx.load_model(f"{model_dir}/segment_model.onnx")).to('cuda')
    print("Onnx model loaded")
    return best_model


def predict_bacteria_cell_mask(img_input, segment_model):
    """predict mask image from input image with DL model provided
    :param img_input: input 2D image
    :param segment_model: Keras DL model for segmentation
    """
    import tensorflow as tf
    img_height, img_width = img_input.shape[:2]
    x_test = np.zeros((1, img_height, img_width), dtype=np.float32)
    x_test[0] = np.squeeze(img_input)
    x_test = x_test[:, :, :, np.newaxis]

    # Predict mask with DL model
    with tf.device('/cpu:0'):
        y = segment_model.predict(x_test)

    # cleanup resulting 2D mask image
    th = 0.5
    y[y > th] = 1
    y[y < th] = 0
    out = (y[0, :, :, 0] * 255).astype(np.uint8)
    return out, y


def predict_bacteria_cell_mask_ptch(img_input, segment_model):
    """predict mask image from input image with DL model provided
    :param img_input: input 2D image
    :param segment_model: Keras DL model for segmentation
    """
    import torch
    img_height, img_width = img_input.shape[:2]
    x_test = np.zeros((1, img_height, img_width), dtype=np.float32)
    x_test[0] = np.squeeze(img_input)
    x_test = x_test[:, :, :, np.newaxis]

    # Predict mask with DL model
    print("pytorch model predicting...")
    x_test_tensor = torch.from_numpy(x_test.copy()).to('cuda').float()
    y = segment_model(x_test_tensor).detach().cpu().numpy()

    # cleanup resulting 2D mask image
    th = 0.5
    y[y > th] = 1
    y[y < th] = 0
    out = (y[0, :, :, 0] * 255).astype(np.uint8)
    return out, y



def load_FusionNet_model(model_dir):
    """load trained Fusion-Net model from model_dir folder"""
    import tensorflow as tf
    # Predict mask with DL model
    with tf.device('/cpu:0'):
	    best_model = keras.models.load_model(fr"{model_dir}/classify_model",
                                        custom_objects={
											 'mean_iou_loss': mean_iou_loss,
											 'mean_iou': mean_iou,
											 'dice_coef': dice_coef,
										 })
    return best_model


def min_max_normalize(img):
    """Normalize data with its min and max
     Args:
          img: 2D or 3D array
     Output:
          2D or 3D array
    """
    out = (img - img.min()) / (img.max() - img.min())
    return out


def d_resize(image, max_dim=64):  # to resize image
    """resize image to 64x64 by padding"""
    resized_img = []
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    image_new = np.pad(image, padding, mode='constant', constant_values=0)
    return(image_new)


def read_image_data_from_table(table):
    from tifffile import imread
    nrows = table.shape[0]
    X = []
    print("reading image data...")
    for i in range(nrows):
        img_name = rf"{table.iat[i, 4].strip()}/{table.iat[i, 5].strip()}"
        img_data = np.squeeze(imread(img_name))
        if img_data is not None:
            # print(img_name)
            # print(img_data.shape)
            img_data = min_max_normalize(img_data)
            img_data = d_resize(img_data, 128)
            X.append(img_data)
    print("reading image data done")
    X = np.asarray(X)
    return X[:, :, :, np.newaxis]


def get_data_for_FusionNet(spec_scaler, shape_scaler, data_dir):
    """collect all information from data folder and form dataset
        for classification with Fusion-Net
        :param: spec_scaler: scaler for spectra
        :param: shape_scaler: scaler for shape statistics
        :param: data_dir: data folder containing all necessary information
        :Returns
            [x_all_spec, x_all_shape, x_all_image]: all three forms of fusion data
    """
    # get main table from CSV file
    import pandas as pd
    filename = fr"{data_dir}/single_cells.csv"
    table = pd.read_csv(filename)
    nptable = table.to_numpy()
    # get all spectra
    x_all_spec = nptable[:, 17:(17+175)].astype(np.float)
    # preprocess
    x_all_spec = snv(x_all_spec)
    # this is preprocessing and should be out of this function
    x_all_spec = np.expand_dims(x_all_spec, axis=2)
    # print(spec_scaler.mean_.shape)
    # print(x_all_spec.shape)
    x_all_spec[:, :, 0] = spec_scaler.transform(x_all_spec[:, :, 0])

    # get all shape
    x_all_shape = nptable[:, 8:17].astype(np.float)
    x_all_shape = shape_scaler.transform(x_all_shape)
    x_all_shape = np.expand_dims(x_all_shape, axis=1)

    # get all image data
    x_all_image = read_image_data_from_table(table)
    return [x_all_spec, x_all_shape, x_all_image]


def classify_data(model_dir, data_dir):
    """Get model & prepare data and then classify single-cell bacteria
    :param: model_dir: folder path containing trained DL model files
    :param: data_dir: folder path containing data for classification
    :Returns
        y: indices of species classified from DL
        yprob: probability of y
    """
    # get model & prepare data for the model
    import pandas as pd
    import tensorflow as tf
    from joblib import load

    spec_scaler = load(fr'{model_dir}/spec_scaler.bin')
    shape_scaler = load(fr'{model_dir}/shape_scaler.bin')
    classify_model = load_FusionNet_model(model_dir)
    x_wide_all = get_data_for_FusionNet(spec_scaler, shape_scaler, data_dir)

    # Predict bacteria with DL model
    with tf.device('/cpu:0'):
        yprobs = classify_model.predict(x_wide_all)
        y = yprobs.argmax(1)
        yprob = yprobs.max(1)

    # save the classification result to output spreadsheet
    filename = fr"{data_dir}/single_cells.csv"
    table = pd.read_csv(filename)
    table["Classified"] = y
    table["Classified_prob"] = yprob
    table.to_csv(filename, index=False)
    return y, yprob


def normalize_spectra_data(input_spectra):
    '''
        Normalize spectral data
        Args:
            spectra: list of spectral data
        Returns:
            List[float]: normalized spectral data
    '''
    data_snv = np.zeros_like(input_spectra)
    for i in range(input_spectra.shape[0]):
        data_snv[i, :] = (input_spectra[i, :] - np.mean(input_spectra[i, :])) / np.std(input_spectra[i, :], ddof=1)
    return data_snv
def read_spectra_data(csv_filename):
    '''
        Read spectral data from CSV file
        Args:
            csv_filename: filename of CSV file containing spectral data
        Returns:
            np.ndarray: spectral data (303,)
    '''

    table = pd.read_csv(csv_filename).to_numpy()
    spectra = table[:, 17:-1].astype(np.float64)
    return spectra

    
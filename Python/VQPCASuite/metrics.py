import numpy as np
from sklearn.metrics import mean_squared_error

def ssim(x, y, c1=1e-10, c2=1e-10):
    '''
    This function calculates the structural similarity index between two arrays
    
    Parameters
    ----------
    x, y : 
        arrays (n,) shape must be consistent
    
    Other parameters
    ----------------
    c1, c2 : float
        constants used for numerical stability
        
    Returns
    -------
    SSIM : float
        Structural Similarity Index (bunded between -1 and 1)
    '''

    # Get shape
    N = len(x)
    if N != len(y):
        raise ValueError("Dimensions between arrays must be consistent")

    # Get mean
    mux = np.mean(x)
    muy = np.mean(y)

    # Get first metric
    lxy = (2*mux*muy + c1)/(mux**2 + muy**2 + c1)   # luminance comparison

    # Get variances
    sx = np.var(x)
    sy = np.var(y)

    # Contrast comparison
    cxy = (2*sx*sy + c2) / (sx**2 + sy**2 + c2)

    # Calculate covariance
    sxy = (1/(N-1)) * np.sum((x-mux)*(y-muy))

    # Calculate similarity index
    c3 = c2/2

    # Similarity index
    SSIM = (2*mux*muy + c1) *  (2*sxy + c2) / ((mux**2 + muy**2 + c1) * (sx**2 + sy**2 + c2))

    return SSIM

def psnr(x, y):

    '''
    This function computes the Peak Signal To Noise Ratio between two arrays

    Parameters
    ----------
    x, y = arrays (n,) shape must be consistent

    Returns
    -------
    PSNR = Peak signal to noise ratio (dB)
    '''

    mxy = max(x, y)
    PSNR = -10 * np.log10(mxy**2 / mean_squared_error(x,y))

    return PSNR

def NMSE(x, y):
    '''
    This function calculates the normalized Mean Squared Error
    
    Parameters
    ----------
    x, y : arrays
        Shape must be consistent
        
    Returns
    -------
    nmse : float
        Normalized Mean Squared Error
    '''

    # Get shape
    N = len(x)
    if N != len(y):
        raise ValueError("Dimensions between arrays must be consistent")
    
    # Sum Of Squared Errors
    sse = np.sum((x-y)**2) 
    # Squares
    sx = np.sum(x**2)
    # Normalized Mean Squared Error
    nmse = sse / sx

    return nmse



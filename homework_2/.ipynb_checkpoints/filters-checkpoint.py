import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = np.pad(image, Hk // 2)
    
    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[i, j] += image[i + k, j + l] * kernel[Hk - 1 - k, Wk - 1 - l]

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    
    npad = ((pad_height, pad_height), (pad_width, pad_width))
    out = np.pad(image.copy(), pad_width=npad, mode='constant', constant_values=0)
    
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    out = np.zeros((Hi, Wi))
    img = zero_pad(image, Hk//2, Wk//2)
    
    kernel = np.flip(kernel, axis=1)
    kernel = np.flip(kernel, axis=0)
            
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(kernel * img[i : i + Hk, j : j + Wk])

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    img = zero_pad(image, Hk//2, Wk//2)
    krnl = zero_pad(kernel, Hk//2, Wk//2)
    # Make the kernel the same size as the padded image
    krnl = np.pad(krnl, ((0, img.shape[0] - krnl.shape[0]), (0, img.shape[1] - krnl.shape[1])), 'constant')

    f_image = np.fft.fft2(img)
    f_kernel = np.fft.fft2(krnl)
    f_output = f_image * f_kernel

    out = np.fft.ifft2(f_output)
    out = np.real(out)

    out = out[Hk // 2:Hi + Hk // 2, Wk // 2:Wi + Wk // 2] # Crop to original image size

    return out

def cross_correlation(f, g0):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    g = g0.copy()
    g = np.flip(g.copy(), axis=0)
    g = np.flip(g.copy(), axis=1)
    
    out = conv_fast(f, g)

    return out

def zero_mean_cross_correlation(f, g0):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    g = g0.copy()
    g = g.astype(float) - np.mean(g)
    g = g.astype(int)
    
    out = cross_correlation(f, g)

    return out
    
def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    f = zero_pad(f, Hg // 2, Wg // 2)
    std_g = np.std(g)
    mean_g = np.mean(g)
    
    for i in range(Hf):
        for j in range(Wf):
            std_f = np.std(f[i : i + Hg, j : j + Wg])
            mean_f = np.mean(f[i : i + Hg, j : j + Wg])
            out[i, j] = np.sum((g - mean_g) * (f[i : i + Hg, j : j + Wg] - mean_f)) / (std_f * std_g)

    return out

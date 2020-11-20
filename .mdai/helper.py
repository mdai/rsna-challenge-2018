import cv2
import keras
import pydicom
import numpy as np
from skimage import exposure


def histogram_equalize(img):
    """ Apply histogram equalization into an image (not used)
    """
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)


def convert_image_to_uint8(img, is_dicom=True):
    """Converts an image to 8bit dtype and monochrome2 (if dicom supplied).
    @img: numpy array or pydicom object -> Represents input image
    @is_dicom: bool -> Represents whether image supplied is dicom or not
    """
    monochrome1 = False
    if is_dicom:
        scale_slope = 1
        scale_intercept = 0
        if "RescaleSlope" in img:
            scale_slope = int(img.RescaleSlope)
        if "RescaleIntercept" in img:
            scale_intercept = int(img.RescaleIntercept)
        if img.PhotometricInterpretation == "MONOCHROME1":
            monochrome1 = True

        img = img.pixel_array
        img = img + scale_intercept
        img = img * scale_slope

    if img.dtype != np.uint8:
        img = (img - img.min()) / (img.max() - img.min())
        img *= 255.0
        img = np.uint8(img)

    if monochrome1:
        img = 255 - img

    return img


def read_image_dicom(path, mode="image"):
    """ Read an image in dicom format.
    Args
        path: Path to the image.
        mode: image|image_sex_view
    """
    dicom_img = pydicom.dcmread(path)
    image = convert_image_to_uint8(dicom_img)
    # convert grayscale to rgb
    image = np.stack((image,) * 3, -1)
    if mode == "image_sex_view":
        # split image in patient sex
        if dicom_img.PatientSex == "F":
            image[:, :, 1] = 0
        elif dicom_img.PatientSex == "M":
            image[:, :, 1] = 1
        else:
            raise Exception("Invalid Sex on dicom {}.".format(path))
        # split image in view position
        if dicom_img.ViewPosition == "AP":
            image[:, :, 2] = 0
        elif dicom_img.ViewPosition == "PA":
            image[:, :, 2] = 1
        else:
            raise Exception(
                "Invalid View Position on dicom {}. View position is: {}".format(
                    path, dicom_img.ViewPosition
                )
            )
    return image[:, :].copy(), dicom_img


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.
    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.
    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    return scale


def preprocess_image(x, mode="caffe"):
    """ Preprocess an image by subtracting the ImageNet mean.
    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.
    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
    elif mode == "caffe":
        x -= [103.939, 116.779, 123.68]
    return x


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.
    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.
    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

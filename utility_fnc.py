import numpy as np


def print_elapsed_time(total_time):
    ''' Prints elapsed time in hh:mm:ss format
    '''
    hh = int(total_time / 3600)
    mm = int((total_time % 3600) / 60)
    ss = int((total_time % 3600) % 60)
    
    print("\n** Total Elapsed Runtime: {:0>2}:{:0>2}:{:0>2}".format(hh, mm, ss))


def resize_image(image, size):
    ''' Resize the image where the shortest side is as many pixels as what is provided with the "size" variable,
        keeping the aspect ratio  
    '''
    w, h = image.size

    if h > w:
        # set width to "size" and scale height to keep the aspect ratio
        h = int(max(h * size / w, 1))
        w = int(size)
    else:
        # set height to "size" and scale width to keep the aspect ratio
        w = int(max(w * size / h, 1))
        h = int(size)

    return image.resize((w, h))


def crop_image(image, size):
    ''' Crop out the center portion of the image as big as what is provided with the "size" variable
    '''
    w, h = image.size

    x1 = (w - size) / 2     # where x1 is the leftmost x-coordinate of the center portion
    y1 = (h - size) / 2     # where y1 is the downmost y-coordinate of the center portion

    x2 = x1 + size      # where x2 is the rightmost x-coordinate of the center portion
    y2 = y1 + size      # where y2 is the uppermost y-coordinate of the center portion

    return image.crop((x1, y1, x2, y2))


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    # resize image where the shortest side is 256 pixels
    resized_image = resize_image(image, 256)

    # crop out the center 224x224 portion of the image
    cropped_image = crop_image(resized_image, 224)

    # convert the color channels of images (encoded as integers 0-255) to floats (interval 0-1)
    np_image = np.array(cropped_image) / 255.

    network_means = [0.485, 0.456, 0.406]
    network_std = [0.229, 0.224, 0.225]

    # normalize images
    mean = np.array(network_means)
    std = np.array(network_std)
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the first dimension
    # but it's the third dimension in the PIL image and Numpy array
    np_image = np_image.transpose((2, 0, 1))

    return np_image

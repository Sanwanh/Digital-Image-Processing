import sys
import cv2 as cv
import numpy as np
import matplotlib as plt

def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0, 2], [3, 1]])
    else:
        smaller_matrix = generate_bayer_matrix(n - 1)
        size = 2 ** n
        new_matrix = np.zeros((size, size), dtype=int)
        for i in range(2 ** (n - 1)):
            for j in range(2 ** (n - 1)):
                base_value = 4 * smaller_matrix[i, j]
                new_matrix[i, j] = base_value
                new_matrix[i, j + 2 ** (n - 1)] = base_value + 2
                new_matrix[i + 2 ** (n - 1), j] = base_value + 3
                new_matrix[i + 2 ** (n - 1), j + 2 ** (n - 1)] = base_value + 1
        
        return new_matrix

def generate_thresholds_matrix(bayer_matrix):
    N = bayer_matrix.shape[0]
    thresholds_matrix = np.zeros_like(bayer_matrix, int)

    # TODO:Calculate each bayer matrix element threshold

    return thresholds_matrix

def Ordered_Dithering(img, thresholds_matrix):
    # TODO:Implementing the ordered dithering algorithm

    return Ordered_Dithering_img

def Error_Diffusion(img):
    # TODO:Implementing the error diffusion algorithm

    return Error_Diffusion_img

if __name__ == '__main__':

    img = cv.imread(sys.argv[1])

    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)
    # TODO:Show your picture


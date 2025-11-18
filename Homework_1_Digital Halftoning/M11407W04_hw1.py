import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
    thresholds_matrix = ((bayer_matrix + 0.5) / (N * N)) * 255
    return thresholds_matrix

def Ordered_Dithering(img, thresholds_matrix):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    H, W = gray.shape
    N = thresholds_matrix.shape[0]
    tiled_thresholds = np.tile(thresholds_matrix, (H // N + 1, W // N + 1))[:H, :W]
    Ordered_Dithering_img = np.where(gray >= tiled_thresholds, 255, 0).astype(np.uint8)
    return Ordered_Dithering_img

def Error_Diffusion(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    Error_Diffusion_img = np.zeros_like(gray)

    kernel = [
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16)
    ]

    for y in range(H):
        for x in range(W):
            old = gray[y, x]
            new = 255 if old >= 128 else 0
            Error_Diffusion_img[y, x] = new
            err = old - new

            for dx, dy, w in kernel:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    gray[ny, nx] += err * w

    return Error_Diffusion_img.astype(np.uint8)

if __name__ == '__main__':
    image_path = 'images/Baboon-image.png'

    output_dir = 'output_results'
    os.makedirs(output_dir, exist_ok=True)

    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image from {image_path}")
        sys.exit(1)

    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)

    ordered_img = Ordered_Dithering(img, thresholds_matrix)
    error_img = Error_Diffusion(img)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ordered_img, cmap='gray')
    plt.title('Ordered Dithering')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(error_img, cmap='gray')
    plt.title('Error Diffusion')
    plt.axis('off')

    plt.show()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ordered_path = os.path.join(output_dir, f'ordered_result_{base_name}.png')
    error_path = os.path.join(output_dir, f'error_diffusion_result_{base_name}.png')

    cv.imwrite(ordered_path, ordered_img)
    cv.imwrite(error_path, error_img)

    print(f'Results saved in folder: {output_dir}')

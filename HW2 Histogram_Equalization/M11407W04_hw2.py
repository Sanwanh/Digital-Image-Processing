import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def Global_HE(img):
    # Compute histogram
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    img_he = cv.LUT(img, cdf_normalized.astype(np.uint8))

    # ---- Overlapped Histogram Plot ----
    plt.figure(figsize=(8,4))
    plt.title('Histogram Before & After Global HE')

    # Before (blue)
    plt.hist(img.flatten(), 256, [0,256], color='blue', alpha=0.5, label='Original')

    # After (orange)
    plt.hist(img_he.flatten(), 256, [0,256], color='orange', alpha=0.5, label='Equalized')

    plt.legend()
    plt.tight_layout()
    plt.show()

    return img_he


def Local_HE(img, tile_size=32):
    h, w = img.shape
    out = np.zeros_like(img)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            hist, _ = np.histogram(tile.flatten(), 256, [0,256])
            cdf = hist.cumsum()
            cdf = cdf * 255 / cdf[-1] if cdf[-1] != 0 else cdf
            tile_he = cv.LUT(tile, cdf.astype(np.uint8))
            out[y:y+tile_size, x:x+tile_size] = tile_he

    return out


if __name__ == '__main__':

    img_path = r"D:\github\Digital-Image-Processing\HW2 Histogram_Equalization\images\F-16-image.png"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("[ERROR] Could not read image:", img_path)
        sys.exit(1)

    global_img = Global_HE(img)
    local_img = Local_HE(img)

    # Show images
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title('Original'); plt.imshow(img, cmap='gray')
    plt.subplot(1,3,2); plt.title('Global HE'); plt.imshow(global_img, cmap='gray')
    plt.subplot(1,3,3); plt.title('Local HE'); plt.imshow(local_img, cmap='gray')
    plt.tight_layout()
    plt.show()

import sys
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Global_DCT(img):
    f = np.float32(img)
    d = cv2.dct(f)
    r = cv2.idct(d)
    return d, np.clip(r,0,255).astype(np.uint8)

def Local_DCT(img, kernel_size=8):
    h, w = img.shape
    img_f = np.float32(img)
    dct_img = np.zeros((h, w), np.float32)
    rec = np.zeros((h, w), np.float32)
    for i in range(0, h, kernel_size):
        for j in range(0, w, kernel_size):
            b = img_f[i:i+kernel_size, j:j+kernel_size]
            d = cv2.dct(b)
            dct_img[i:i+kernel_size, j:j+kernel_size] = d
            r = cv2.idct(d)
            rec[i:i+kernel_size, j:j+kernel_size] = r
    return dct_img, np.clip(rec,0,255).astype(np.uint8)

def frequency_Domain_filter(DCT_img, keep=30):
    h, w = DCT_img.shape
    filtered = np.zeros((h, w), np.float32)
    filtered[:keep, :keep] = DCT_img[:keep, :keep]
    rec = cv2.idct(filtered)
    return filtered, np.clip(rec,0,255).astype(np.uint8)

def extract_blocks(img, block_size=4):
    h, w = img.shape
    new_h = math.ceil(h / block_size) * block_size
    new_w = math.ceil(w / block_size) * block_size

    padded = np.zeros((new_h, new_w), np.float32)
    padded[:h, :w] = img

    blocks = []
    for i in range(0, new_h, block_size):
        for j in range(0, new_w, block_size):
            b = padded[i:i+block_size, j:j+block_size].flatten()
            blocks.append(b)

    return np.array(blocks, dtype=np.float32), (new_h, new_w)

def lbg_codebook_training(vectors, codebook_size=64, epsilon=1e-3, max_iter=100):
    idx = np.random.choice(len(vectors), codebook_size, replace=False)
    codebook = vectors[idx].copy()
    for _ in range(max_iter):
        dist = np.sum((vectors[:,None,:] - codebook[None,:,:])**2, axis=2)
        labels = np.argmin(dist, axis=1)
        new_cb = np.zeros_like(codebook)
        for k in range(codebook_size):
            cluster = vectors[labels == k]
            if len(cluster) > 0:
                new_cb[k] = np.mean(cluster, axis=0)
            else:
                new_cb[k] = codebook[k]
        diff = np.linalg.norm(new_cb - codebook)
        codebook = new_cb
        if diff < epsilon:
            break
    return codebook

def vq_encode(vectors, codebook):
    dist = np.sum((vectors[:,None,:] - codebook[None,:,:])**2, axis=2)
    return np.argmin(dist, axis=1)

def vq_decode(indices, codebook, image_shape, block_size=4):
    h, w = image_shape
    rec = np.zeros((h, w), np.float32)
    c = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            v = codebook[indices[c]].reshape(block_size, block_size)
            rec[i:i+block_size, j:j+block_size] = v
            c += 1
    return np.clip(rec,0,255).astype(np.uint8)

def visualize_codebook_as_table(codebook, block_size=4):
    df = pd.DataFrame(codebook)
    plt.figure(figsize=(10,6))
    plt.title("Codebook Table")
    plt.imshow(df, cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("codebook_table.png")

if __name__ == '__main__':
    img = cv2.imread(r"D:\github\Digital-Image-Processing\HW3 DCT_VQ\images\Baboon-image.png", cv2.IMREAD_GRAYSCALE)

    dct_g, rec_g = Global_DCT(img)
    dct_l, rec_l = Local_DCT(img)
    fd_f, fd_rec = frequency_Domain_filter(dct_g)

    vectors, shape = extract_blocks(img, 4)
    codebook = lbg_codebook_training(vectors, 64)
    indices = vq_encode(vectors, codebook)
    vq_rec = vq_decode(indices, codebook, shape, 4)
    visualize_codebook_as_table(codebook)

    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1); plt.title("Original"); plt.imshow(img, cmap='gray')
    plt.subplot(2,3,2); plt.title("Global DCT"); plt.imshow(np.log(np.abs(dct_g)+1), cmap='gray')
    plt.subplot(2,3,3); plt.title("IDCT Global"); plt.imshow(rec_g, cmap='gray')
    plt.subplot(2,3,4); plt.title("Filtered DCT"); plt.imshow(np.log(np.abs(fd_f)+1), cmap='gray')
    plt.subplot(2,3,5); plt.title("Filtered Rec"); plt.imshow(fd_rec, cmap='gray')
    plt.subplot(2,3,6); plt.title("VQ Rec"); plt.imshow(vq_rec, cmap='gray')
    plt.tight_layout()
    plt.show()

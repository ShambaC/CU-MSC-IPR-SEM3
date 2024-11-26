import cv2
import numpy as np
import click

from matplotlib import pyplot as plt

@click.command()
@click.option('--file', '-F', help='Absolute location of the image file')
def main(file) :
    img = cv2.imread(file, 0)
    img_eq = img.copy()

    total_pixels = img.shape[0] * img.shape[1]
    L = 256

    # Calculate histogram
    level, count = np.unique(img, return_counts=True)
    hist_og = np.asarray((level, count)).T
    
    def pr_calc(row : np.ndarray) -> float :
        pr  = row[1] / total_pixels
        return pr
    
    hist_og_pr = np.apply_along_axis(pr_calc, 1, hist_og)
    hist_og_pr = hist_og_pr.reshape((hist_og_pr.shape[0], 1))
    hist_og_pr = np.hstack([hist_og, hist_og_pr])

    hist_eq = hist_og.copy()
    for i in range(hist_og.shape[0]) :
        hist_eq[i][0] = int((L - 1) * hist_og_pr[:i+1, -1:].sum().item())

    for i in range(hist_og.shape[0]) :
        img_eq[img == hist_og[i][0]] = hist_eq[i][0]

    level_eq, count_eq = np.unique(img_eq, return_counts=True)

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_eq, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.bar(level, count)
    plt.xlabel("Levels")
    plt.ylabel("Count")
    plt.title("Original histogram")

    plt.subplot(2, 2, 4)
    plt.bar(level_eq, count_eq)
    plt.xlabel("Levels")
    plt.ylabel("Count")
    plt.title("equalized histogram")

    plt.show()

if __name__ == '__main__' :
    main()
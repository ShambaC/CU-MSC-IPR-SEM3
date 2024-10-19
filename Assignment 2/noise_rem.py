import cv2
import numpy as np
import click    

@click.command()
@click.option('--file', '-F', help='Absolute location of the image file')
@click.option('--num','-N', default=10, help='Number of noisy images')
def main(file, num) :

    img = cv2.imread(file)
    h, w, _ = img.shape
    
    mu = 0
    sigma = 0.1

    num_imgs = num

    noisy_list = []

    for i in range(num_imgs) :
        noisy_img = img.copy()
        
        gaussian_noise = np.random.normal(mu, sigma, noisy_img.shape)
        gaussian_noise = gaussian_noise * 255
        gaussian_noise = gaussian_noise.astype(np.uint8)
        
        noisy_img += gaussian_noise
        noisy_img = noisy_img.astype(np.uint8)
        
        noisy_list.append(noisy_img)

    avg_mat = np.zeros((h, w, 3))
    for i in range(num_imgs) :
        avg_mat = np.add(avg_mat, noisy_list[i])

    avg_mat = np.divide(avg_mat, num_imgs)
    avg_mat = avg_mat.astype(np.uint8)

    while True :
        # for i in range(num_imgs) :
        #     cv2.imshow(f"noise{i}", noisy_list[i])
        
        cv2.imshow("NoiseGone", avg_mat)

        if cv2.waitKey(1) == ord('q') :
            cv2.destroyAllWindows()
            break


if __name__ == '__main__' :
    main()
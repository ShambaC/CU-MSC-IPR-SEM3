import cv2
import numpy as np
import click    

@click.command()
@click.option('--file', '-F', help='Absolute location of the image file')
def main(file) :

    img = cv2.imread(file)
    h, w, _ = img.shape
    
    pepper = 0.02
    salt = 1-0.02

    num_imgs = 5

    prob_mat = np.random.random((h, w))
    noisy_list = []

    for i in range(num_imgs) :
        noisy_img = img.copy()

        pepper_mask = prob_mat < pepper
        salt_mask = prob_mat > salt

        noisy_img[pepper_mask] = 0
        noisy_img[salt_mask] = 255

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
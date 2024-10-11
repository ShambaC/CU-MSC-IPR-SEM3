import cv2
import numpy as np
import click

@click.command()
@click.option('--file', '-F', help='Absolute location of the image file')
@click.option('--gamma', '-G', default=2.0, help='Gamma value')
def main(file, gamma) :

    img = cv2.imread(file)

    gamma_img = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

    while True :
        
        cv2.imshow("Original", img)
        cv2.imshow("Gamma", gamma_img)

        if cv2.waitKey(1) == ord('q') :
            cv2.destroyAllWindows()
            break

if __name__ == '__main__' :
    main()
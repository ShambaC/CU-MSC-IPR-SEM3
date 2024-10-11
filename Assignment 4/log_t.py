import cv2
import numpy as np
import click

@click.command()
@click.option('--file', '-F', help='Absolute location of the image file')
def main(file) :

    img = cv2.imread(file)

    c = 255 / np.log(1 + 255)

    res = c * np.log(np.add(img, 1))
    res = res.astype(np.uint8)

    while True :
        
        cv2.imshow("Original", img)
        cv2.imshow("Log transform", res)

        if cv2.waitKey(1) == ord('q') :
            cv2.destroyAllWindows()
            break

if __name__ == '__main__' :
    main()
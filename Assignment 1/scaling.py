import cv2
import numpy as np
import click

def nearest(src : cv2.typing.MatLike, scale : int = 2) -> np.ndarray :
    """ Method to scale image using the nearest neighbour algorithm

        Parameters
        ----------
        src : MatLike
            Image to scale
        scale : int
            Amount to scale, default 2

        Returns
        -------
        ndarray
            scaled image
    """

    h, w, c = src.shape
    res = np.empty((h  * scale, w  * scale, c), np.float32)
    
    with np.nditer(res, flags=['multi_index'], op_flags=['readwrite']) as it :
        for pixel in it :
            x, y, z = it.multi_index
            pixel[...] = src[int(x / scale), int(y / scale), int(z / scale)]

    return res


def bilinear(src : cv2.typing.MatLike, scale : int = 2) :
    pass

@click.command()
@click.option('--file', '-F', help='Absolute location of the image file')
@click.option('--type', '-T', type=click.Choice(['nearest', 'bilinear'], case_sensitive=False), help='Type of scaling, nearest neighbour or Bilinear')
@click.option('--scale', '-S', default=2, type=click.INT, help='Scale factor of the image')
def main(file, type, scale) :
    """A program to scale image"""
    
    src = cv2.imread(file)
    res = None

    if type == 'nearest' :
        res = nearest(src, scale)

    while True :
        cv2.imshow("Input", src)
        cv2.imshow("Output", res)

        if cv2.waitKey(1) == ord('q') :
            cv2.destroyAllWindows()
            break


if __name__ == '__main__' :
    main()


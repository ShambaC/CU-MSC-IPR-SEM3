import cv2
import numpy as np
import click

def nearest(src : cv2.typing.MatLike, scale : any = 2) -> np.ndarray :
    """ Method to scale image using the nearest neighbour algorithm

        Parameters
        ----------
        src : MatLike
            Image to scale
        scale : any
            Amount to scale, default 2

        Returns
        -------
        ndarray
            scaled image
    """

    h, w, c = src.shape

    
    # For loop method
    # res = np.empty((int(h  * scale), int(w  * scale), c), np.uint8)
    
    # with np.nditer(res, flags=['multi_index'], op_flags=['readwrite']) as it :
    #     for pixel in it :
    #         x, y, z = it.multi_index
    #         pixel[...] = src[int(x / scale), int(y / scale), z]

    # return res
    

    # Vectorized method
    new_h, new_w = int(h * scale), int(w * scale)
    row, col = np.indices((new_h, new_w))

    x_src = (row / scale).astype(int)
    y_src = (col / scale).astype(int)

    return src[x_src, y_src]


def bilinear(src : cv2.typing.MatLike, scale : int = 2) :
    """ Method to scale image using the bilinear interpolation algorithm

        Parameters
        ----------
        src : MatLike
            Image to scale
        scale : any
            Amount to scale, default 2

        Returns
        -------
        ndarray
            scaled image
    """
    import math

    h, w, c = src.shape
    res = np.empty((int(h  * scale), int(w  * scale), c), np.uint8)

    with np.nditer(res, flags=['multi_index'], op_flags=['readwrite']) as it :
        for pixel in it :
            x, y, z = it.multi_index

            x_l, y_l = math.floor(y / scale), math.floor(x / scale)
            x_h, y_h = math.ceil(y / scale) if math.ceil(y / scale) < w else w - 1, math.ceil(x / scale) if math.ceil(x / scale) < h else h - 1

            # print(f'({x}, {y}, {z})=>[{x_l}, {y_l}, {x_h}, {y_h}]')

            x_weight = (scale * y) - x_l
            y_weight = (scale * x) - y_l

            a = src[y_l, x_l, z]
            b = src[y_l, x_h, z]
            c = src[y_h, x_l, z]
            d = src[y_h, x_h, z]

            pixel[...] = a * (1 - x_weight) * (1 - y_weight) \
                         + b * x_weight * (1 - y_weight) + \
                         c * y_weight * (1 - x_weight) + \
                         d * x_weight * y_weight

    return res 


@click.command()
@click.option('--file', '-F', help='Absolute location of the image file')
@click.option('--type', '-T', type=click.Choice(['nearest', 'bilinear'], case_sensitive=False), help='Type of scaling, nearest neighbour or Bilinear')
@click.option('--scale', '-S', default=2.0, help='Scale factor of the image')
def main(file, type, scale) :
    """A program to scale image"""
    
    src = cv2.imread(file)
    res = None

    if type == 'nearest' :
        res = nearest(src, scale)
    elif type == 'bilinear' :
        res = bilinear(src, scale)

    while True :
        cv2.imshow("Input", src)
        cv2.imshow("Output", res)

        if cv2.waitKey(1) == ord('q') :
            cv2.destroyAllWindows()
            break


if __name__ == '__main__' :
    main()


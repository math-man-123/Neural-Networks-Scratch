# pyright: reportMissingImports=false, reportUndefinedVariable=false

import numpy as np
from scipy.ndimage import affine_transform
from factory import factory, make
from typing import Any
from custom_types import *


class Augmentor:
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Create a new Augmentor instance with given config.
        
        :param config: augmentor factory configs
        """
        self._augmentor = make(config)

    def create(self, dataset: list[Array, Vector]) -> list[Array, Vector]:
        """
        Apply transformation to each element in dataset 
        and return set of new transformed elements.
        
        :param dataset: data to transform with labels
        :return: new dataset of transformed data
        """
        return self._augmentor(dataset)


@factory
def make_affine2D(
    img_shape: tuple[int] = (28, 28),
    min_val: float = -0.5,
    max_val:float = 0.5,
    augment_factor: float = 1.35,
    standard_factor: float = 2.58,
    max_rotate: float = 30.0,
    max_shear: float = 0.7,
    max_scale: float = 0.4,
    max_trans: float = 5.0
):

    def normal(max_val) -> float:
        """
        Get a normal distributed random value in
        [+- max_val] most of the time. Change the
        quantile standard_factor to affect what
        that means e.g. 2.58 is 99% of the time.
        
        :param max_val: maximum value
        """
        deviation = max_val / standard_factor
        return deviation * np.random.normal()

    def rotate() -> Matrix:
        """
        Get a rotation matrix for a random angle.
        """
        theta = np.deg2rad(normal(max_rotate))

        rot_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]])
        
        return rot_matrix

    def shear() -> Matrix:
        """
        Get a shear matrix for a seperate 
        random shear in both x and y.
        """
        shear_x = normal(max_shear)
        shear_y = normal(max_shear)

        shear_matrix = np.array(
            [[1, shear_y], 
             [shear_x, 1]])

        return shear_matrix

    def scale() -> Matrix:
        """
        Get a scale matrix for a seperate 
        random scale in both x and y.
        """
        scale_x = normal(max_scale)
        scale_y = normal(max_scale)

        scale_matrix = np.array(
            [[1 + scale_x, 0],
             [0, 1 + scale_y]])

        return scale_matrix

    def translate() -> Vector:
        """
        Get a translation vector for a seperate
        random translation in both x and y.
        """
        trans_x = normal(max_trans)
        trans_y = normal(max_trans)

        trans_vect = np.array([trans_x, trans_y])
        
        return trans_vect

    def affine2D(dataset: list[Array, Vector]) -> list[Array, Vector]:
        """
        Apply a affine transformation Ax + b to every element
        of given dataset. A = RHS where R is a rotation, H is
        a shear, and S is a scale. Order matters here!
        
        :param dataset: data to transform with labels
        :return: new dataset of transformed data
        """
        data_shape = dataset[0][0].shape
        transform = []
        for data in dataset:
            img = data[0].reshape(img_shape)
            matrix = rotate() @ shear() @ scale()

            # invert and swap (x,y) <-> (row,col) for the
            # correct use of scipy.affine_transformation
            matrix = np.linalg.inv(matrix)
            swap = np.array([[0, 1], [1, 0]]) 
            matrix = swap @ matrix @ swap

            # calculate image center offset
            h, w = img_shape
            center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
            offset = center - matrix @ (center + translate())

            # apply affine transformation to img
            augment = affine_transform(
                input=img, matrix=matrix, offset=offset, order=1, cval=min_val)
            
            # increase augment data by given factor
            augment = np.clip(augment * augment_factor, min_val, max_val)
            transform.append((augment.reshape(data_shape), data[1]))

        return transform
    
    return affine2D

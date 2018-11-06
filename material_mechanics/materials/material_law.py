#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module holds possible material laws
"""
import logging
import numpy as np

from ..tools import force_symmetry


# ====================
# Set up module logger
# ====================
logger = logging.getLogger(__name__)


class HookesLaw:
    r"""
    representation of Hookes Law of elasticity for two- and three-dimensional stress states in materials

    .. note::
        poisson ratios are in international notation.
        The first index points to the causing strain.
        The second index points to the resulting strain.

    :param stiffness: List or tuples of pairs of stiffness values. Each pair holds the stiffness in a major axis of the
        material and the shear stiffness in the plane perpendicular to that axis.
        ((:math:`\sigma_{11}`, :math:`G_{23}`) for the 1-direction)
    :param poisson: List or tuples of pairs of poisson ratios. each pair holds the two poisson ratios perpendicular to a
        major material axis. So the first entry holds (:math:`\nu_{23}`, :math:`\nu_{32}`).
    :type stiffness: List or tuple of lists or tuples of floats
    :type poisson: List or tuple of lists or tuples of floats
    """

    def __init__(self, stiffness, poisson, *args, **kwargs):
        self.symmetry = kwargs.get('symmetry', 'mean')

        self.e1 = stiffness[0][0]
        self.e2 = stiffness[1][0]
        self.e3 = stiffness[2][0]

        self.g23 = stiffness[0][1]
        self.g13 = stiffness[1][1]
        self.g12 = stiffness[2][1]

        self.nu23 = poisson[0][0]
        self.nu32 = poisson[0][1]
        self.nu13 = poisson[1][0]
        self.nu31 = poisson[1][1]
        self.nu12 = poisson[2][0]
        self.nu21 = poisson[2][1]

    @property
    def compliance_matrix(self):
        """
        returns the compliance matrix for three-dimensional stress states

        :return: compliance matrix (6x6)
        :rtype: numpy.ndarray
        """
        compliance_matrix = np.zeros((6, 6))
        compliance_matrix[0, 0] = 1. / self.e1
        compliance_matrix[0, 1] = -self.nu21 / self.e2
        compliance_matrix[0, 2] = -self.nu31 / self.e3
        compliance_matrix[1, 0] = -self.nu12 / self.e1
        compliance_matrix[1, 1] = 1. / self.e2
        compliance_matrix[1, 2] = -self.nu32 / self.e3
        compliance_matrix[2, 0] = -self.nu13 / self.e1
        compliance_matrix[2, 1] = -self.nu23 / self.e2
        compliance_matrix[2, 2] = 1. / self.e3
        compliance_matrix[3, 3] = 1. / self.g23
        compliance_matrix[4, 4] = 1. / self.g13
        compliance_matrix[5, 5] = 1. / self.g12
        return compliance_matrix

    @property
    def compliance_matrix_2d(self):
        """
        returns the compliance matrix for two-dimensional stress states

        :return: compliance matrix (3x3)
        :rtype: numpy.ndarray
        """
        compliance_matrix = np.zeros((3, 3))
        compliance_matrix[0, 0] = 1. / self.e1
        compliance_matrix[0, 1] = -self.nu21 / self.e2
        compliance_matrix[1, 0] = -self.nu12 / self.e1
        compliance_matrix[1, 1] = 1. / self.e2
        compliance_matrix[2, 2] = 1. / self.g12
        return compliance_matrix

    @property
    def stiffness_matrix(self):
        """
        returns the stiffness matrix for three-dimensional stress states

        :return: stiffness matrix (6x6)
        :rtype: numpy.ndarray
        """
        return np.linalg.inv(self.compliance_matrix)

    @property
    def stiffness_matrix_2d(self):
        """
        returns the stiffness matrix for two-dimensional stress states

        :return: stiffness matrix (3x3)
        :rtype: numpy.ndarray
        """
        return np.linalg.inv(self.compliance_matrix_2d)

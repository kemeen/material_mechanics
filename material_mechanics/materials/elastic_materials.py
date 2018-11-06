#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from material_mechanics.materials.material_law import HookesLaw

# ====================
# Set up module logger
# ====================
logger = logging.getLogger(__name__)


class ElasticMaterial(object):
    r"""
    Creates an elastic material

    :param stiffness: stiffness values of the material in :math:`\frac{N}{mm^2}`
    :param poisson: poisson ratios of the material
    :param strength: (*optional*) strength values of the material in :math:`\frac{N}{mm^2}`, default is None
    :param density: (*optional*) material density in :math:`\frac{g}{cm^3}`, default is None
    :type stiffness: list
    :type poisson: list
    :type strength: list
    :type density: float

    Example::

        import material_mechanics as mm

        stiffness = [(10000, 3000),(8000, 2500),(5000, 1800)]
        poisson = [(0.3, 0.02),(0.28, 0.02),(0.34, 0.34)]

        mat = mm.materials.elastic_materials.ElasticMaterial(stiffness=stiffness, poisson=poisson)

    This class is not available at the top level since the use of the factory functions is encouraged. The same result
    as in the above example can be achieved by using the function
    :func:`orthotropic_material() <material_mechanics.materials.material_factories.orthotropic_material>`

    """

    def __init__(self, stiffness, poisson, *args, **kwargs):
        self._stiffness = stiffness
        self._poisson = poisson
        self._strength = kwargs.get('strength', None)
        self._density = kwargs.get('density', None)
        self.name = kwargs.get('name', None)
        self.symmetry = kwargs.get('symmetry', 'mean')
        self._material_law = HookesLaw(stiffness=stiffness, poisson=poisson, symmetry=self.symmetry)

    def __str__(self):
        repr_str = f'E1: {self.get_stiffness(11)}, E2:{self.get_stiffness(22)}, G12:{self.get_stiffness(12)}, ' \
                   f'nu12:{self.get_poisson(12, precision=4)}, nu21:{self.get_poisson(21, precision=4)}, ' \
                   f'nu23:{self.get_poisson(23, precision=4)}, ' \
                   f'Rlp:{self.get_strength(11, "+")}, Rlm:{self.get_strength(11, "-")}, ' \
                   f'Rpp:{self.get_strength(22, "+")}, Rpp:{self.get_strength(22, "-")}, ' \
                   f'Rpl:{self.get_strength(12)}'
        return repr_str

    @property
    def compliance_matrix(self):
        """
        Return the materials compliance matrix for three-dimensional stress states

        :return: compliance matrix (6x6)
        :rtype: numpy.ndarray
        """
        return self._material_law.compliance_matrix

    @property
    def compliance_matrix_2d(self):
        """
        Return the materials compliance matrix for two-dimensional stress states

        :return: compliance matrix (3x3)
        :rtype: numpy.ndarray
        """
        return self._material_law.compliance_matrix_2d

    @property
    def stiffness_matrix(self):
        """
        Return the materials stiffness matrix for three-dimensional stress states

        :return: stiffness matrix (6x6)
        :rtype: numpy.ndarray
        """
        return self._material_law.stiffness_matrix

    @property
    def stiffness_matrix_2d(self):
        """
        Return the materials stiffness matrix for two-dimensional stress states

        :return: stiffness matrix (3x3)
        :rtype: numpy.ndarray
        """
        return self._material_law.stiffness_matrix_2d

    @property
    def density(self):
        """
        return the material density
        :return: density
        :rtype: float
        """
        return self._density

    def get_stiffness(self, index=None, *args, **kwargs):
        """
        return a stiffness value of the material in the requested direction

        :param index: (*optional*) index of the requested stiffness value, default is None, returning the
            major stiffness in direction '11'

            Options:

                - :math:`E_{11}`: (1, 11, '1', '11', None, 'x', 'X')
                - :math:`E_{22}`: (2, 22, '2', '22', 'y', 'Y')
                - :math:`E_{33}`: (3, 33, '3', '33', 'z', 'Z')
                - :math:`G_{23}`, :math:`G_{32}`: (23, 32, '23', '32', 'yz', 'YZ', 'zy', 'ZY')
                - :math:`G_{13}`, :math:`G_{32}`: (13, 31, '13', '31', 'xz', 'XZ', 'zx', 'ZX')
                - :math:`G_{12}`, :math:`G_{21}`: (12, 21, '12', '21', 'xy', 'XY', 'yx', 'YX')

        :param precision: number of decimal points of the requested stiffness
        :type index: int, str or None
        :type precision: int
        :return: requested stiffness
        :rtype: float

        Example::

            import material_mechanics as mm

            stiffness = [(10000, 3000),(8000, 2500),(5000, 1800)]
            poisson = [(0.3, 0.02),(0.28, 0.02),(0.34, 0.34)]

            mat = mm.materials.elastic_materials.ElasticMaterial(stiffness=stiffness, poisson=poisson)

            e1 = mat.get_stiffness()
            e1 = mat.get_stiffness(11)
            g12 = mat.get_stiffness('12')
        """

        precision = kwargs.get('precision', 2)
        value = None

        if index in (1, 11, '1', '11', None, 'x', 'X'):
            value = self._stiffness[0][0]
        if index in (2, 22, '2', '22', 'y', 'Y'):
            value = self._stiffness[1][0]
        if index in (3, 33, '3', '33', 'z', 'Z'):
            value = self._stiffness[2][0]
        if index in (12, 21, '12', '21', 'xy', 'XY', 'yx', 'YX'):
            value = self._stiffness[2][1]
        if index in (13, 31, '13', '31', 'xz', 'XZ', 'zx', 'ZX'):
            value = self._stiffness[1][1]
        if index in (23, 32, '23', '32', 'yz', 'YZ', 'zy', 'ZY'):
            value = self._stiffness[0][1]
        if value is not None:
            return round(value, precision)
        else:
            return value

    def get_poisson(self, index=None, *args, **kwargs):
        r"""
        return a poisson ratio of the material in the requested direction

        :param index: (*optional*) index of the requested poisson ratio, default is None, returning the
            major poisson ratio in direction :math:`\nu_{12}`

            Options:

                - :math:`\nu_{23}`: (23, '23', 'yz', 'YZ')
                - :math:`\nu_{32}`: (32, '32', 'zy', 'ZY')
                - :math:`\nu_{13}`: (13, '13', 'xz', 'XZ')
                - :math:`\nu_{31}`: (31, '31', 'zx', 'ZX')
                - :math:`\nu_{23}`: (12, '12', 'xy', 'XY', None)
                - :math:`\nu_{32}`: (21, '21', 'yx', 'YX')

        :param precision: number of decimal points of the requested poisson ratio
        :type index: int, str or None
        :type precision: int
        :return: requested poisson ratio
        :rtype: float

        .. note::
            poisson ratios are in international notation.
            The first index points to the causing strain.
            The second index points to the resulting strain.

        Example::

            import material_mechanics as mm

            stiffness = [(10000, 3000),(8000, 2500),(5000, 1800)]
            poisson = [(0.3, 0.02),(0.28, 0.02),(0.34, 0.34)]

            mat = mm.materials.elastic_materials.ElasticMaterial(stiffness=stiffness, poisson=poisson)

            nu12 = mat.get_poisson()
            nu12 = mat.get_poisson(12)
            nu32 = mat.get_poisson('32')
        """

        precision = kwargs.get('precision', 2)

        if index in (23, '23', 'yz', 'YZ'):
            return round(self._poisson[0][0], precision)
        if index in (32, '32', 'zy', 'ZY'):
            return round(self._poisson[0][1], precision)
        if index in (13, '13', 'xz', 'XZ'):
            return round(self._poisson[1][0], precision)
        if index in (31, '31', 'zx', 'ZX'):
            return round(self._poisson[1][1], precision)
        if index in (12, '12', 'xy', 'XY', None):
            return round(self._poisson[2][0], precision)
        if index in (21, '21', 'yx', 'YX'):
            return round(self._poisson[2][1], precision)
        return None

    def get_strength(self, index=None, direction=None, *args, **kwargs):
        """
        return strength value of the material in the requested direction

        :param index: (*optional*) index of the requested stiffness value, default is None, returning the
            major stiffness in direction '11'

            Options:

                - :math:`R_{11}^{+/-}`: (1, 11, '1', '11', None, 'x', 'X')
                - :math:`R_{22}^{+/-}`: (2, 22, '2', '22', 'y', 'Y')
                - :math:`R_{33}^{+/-}`: (3, 33, '3', '33', 'z', 'Z')
                - :math:`R_{23}`: (23, '23', 'yz', 'YZ')
                - :math:`R_{32}`: (32, '32', 'zy', 'ZY')
                - :math:`R_{13}`: (13, '13', 'xz', 'XZ')
                - :math:`R_{31}`: (31, '31', 'zx', 'ZX')
                - :math:`R_{12}`: (12, '12', 'xy', 'XY')
                - :math:`R_{21}`: (21, '21', 'yx', 'YX')

        :param direction: (*optional*) tensile or compression, default is None (resulting in tensile strength values)

            Options:

                - tensile: (1, '1', '+', 'tensile', 't', 'plus', 'p', 'positive', 'pos', None)
                - compression: (-1, '-1', '-', 'compression', 'c', 'minus', 'm', 'negative', 'neg', 'n')
                
        :param precision: (*optional*) number of decimal points of the requested stiffness
        :type index: int, str or None
        :type precision: int or None
        :type direction: str or None
        :return: requested stiffness
        :rtype: float

        Example::

            import material_mechanics as mm

            stiffness = [(10000, 3000),(8000, 2500),(5000, 1800)]
            poisson = [(0.3, 0.02),(0.28, 0.02),(0.34, 0.34)]

            mat = mm.materials.elastic_materials.ElasticMaterial(stiffness=stiffness, poisson=poisson)

            r1 = mat.get_strength()
            r1 = mat.get_strength(11)
            r12 = mat.get_strength('12')
        """

        if self._strength is None:
            return None

        tensile = (1, '1', '+', 'tensile', 't', 'plus', 'p', 'positive', 'pos', None)
        compression = (-1, '-1', '-', 'compression', 'c', 'minus', 'm', 'negative', 'neg', 'n')

        if index in (1, 11, '1', '11', 'x', 'X', None):
            if direction in tensile:
                return self._strength[0][0]
            if direction in compression:
                return self._strength[0][1]

        if index in (2, 22, '2', '22', 'y', 'Y'):
            if direction in tensile:
                return self._strength[1][0]
            if direction in compression:
                return self._strength[1][1]

        if index in (3, 33, '3', '33', 'z', 'Z'):
            if direction in tensile:
                return self._strength[2][0]
            if direction in compression:
                return self._strength[2][1]

        if index in (23, '23', 'yz', 'YZ'):
            return self._strength[0][2]

        if index in (32, '32', 'zy', 'ZY'):
            return self._strength[0][3]

        if index in (13, '13', 'xz', 'XZ'):
            return self._strength[1][2]

        if index in (31, '31', 'zx', 'ZX'):
            return self._strength[1][3]

        if index in (12, '12', 'xy', 'XY'):
            return self._strength[2][2]

        if index in (21, '21', 'yx', 'YX'):
            return self._strength[2][3]

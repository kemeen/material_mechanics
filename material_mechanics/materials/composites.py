#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import logging
import numpy as np
from .elastic_materials import ElasticMaterial
from ..tools import get_T_strain_2d, get_strain_at_z, force_symmetry

# ====================
# Set up module logger
# ====================
logger = logging.getLogger(__name__)


class FiberReinforcedPlastic(ElasticMaterial):
    """
    defines a fiber reinforced material consisting of two constituents, fiber and embedding matrix

    both materials need to be of a material type derived from
    :class:`ElasticMaterial <material_mechanics.materials.elastic_materials.ElasticMaterial>`

    :param fiber_material: fiber material
    :param matrix_material: matrix material
    :param fiber_volume_fraction: volume fraction of fiber material
    :param name: (*optional*) material name. Default is None
    :param strength: (*optional*) material strength values of the composite at the provided fiber volume ratio.
        Default is None
    :type fiber_material: pymat material
    :type matrix_material: pymat material
    :type fiber_volume_fraction: float
    :type name: str
    :type strength: dict of floats

    Example::

        >>> import material_mechanics as mm
        >>>
        >>> # fiber definition
        >>> name = 'Carbon Fiber HT'
        >>> stiffness = dict(e1=230000, e2=13000, g12=50000)
        >>> poisson = dict(nu12=0.23, nu23=0.3)
        >>> fiber = mm.transverse_isotropic_material(name=name, stiffness=stiffness, poisson=poisson, density=1.74)
        >>>
        >>> # matrix definition
        >>> name = 'Epoxy Resin'
        >>> matrix = mm.isotropic_material(name=name, stiffness=3200.0, poisson=0.3, density=1.2)
        >>>
        >>> # fiber volume content
        >>> phi = 0.65
        >>>
        >>> # fiber reinforced material initialization
        >>> frp_material = mm.FiberReinforcedPlastic(
        >>>     fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi
        >>> )
        >>> print(frp_material)
        Name: CarbonFiberHT_EpoxyResin_65, fiber volume fraction: 0.65, Fiber: Carbon Fiber HT, Matrix: Epoxy Resin'
    """

    def __init__(self, fiber_material, matrix_material, fiber_volume_fraction, *args, **kwargs):
        self._fiber_material = fiber_material
        self._matrix_material = matrix_material
        self._fiber_volume_fraction = fiber_volume_fraction
        self.name = kwargs.get('name', None)
        self.symmetry = kwargs.get('symmetry', 'mean')

        if self.name is None:
            fvc_percent = int(round(self.fiber_volume_fraction * 100, 0))
            self.name = f'{self._fiber_material.name}_{self._matrix_material.name}_{fvc_percent}'.replace(" ", "").replace("-", "")

        # density
        try:
            self._density = fiber_material.density * fiber_volume_fraction + matrix_material.density * (
                        1 - fiber_volume_fraction)
        except TypeError:
            logger.exception('Density for {name} could not be calculated. Returning None'.format(name=self.name))
            self._density = None

        e1 = self._get_e1()
        e2 = self._get_e2()
        g12 = self._get_g12()
        nu12 = self._get_nu12()
        nu23 = self._get_nu23()
        nu21 = e2 / e1 * nu12
        g23 = e2 / (2 * (1 + nu23))

        stiffnesses = [(e1, g23), (e2, g12), (e2, g12)]
        poissons = [(nu23, nu23), (nu12, nu21), (nu12, nu21)]
        strength_dict = kwargs.get('strength', None)

        if strength_dict is not None:
            _strengths = [
                (strength_dict.get('r_11_tensile', None), strength_dict.get('r_11_compression', None),
                 strength_dict.get('r_23', None), strength_dict.get('r_23', None)),
                (strength_dict.get('r_22_tensile', None), strength_dict.get('r_22_compression', None),
                 strength_dict.get('r_12', None), strength_dict.get('r_21', None)),
                (strength_dict.get('r_22_tensile', None), strength_dict.get('r_22_compression', None),
                 strength_dict.get('r_12', None), strength_dict.get('r_21', None)),
            ]
        else:
            _strengths = None

        super(FiberReinforcedPlastic, self).__init__(
            stiffness=stiffnesses, poisson=poissons, strength=_strengths, symmetry=self.symmetry, name=self.name
        )

    # def __str__(self):
    #     return self.name

    def __eq__(self, other):
        if not isinstance(other, FiberReinforcedPlastic):
            return False
        ret = [
            self.fiber_material == other.fiber_material,
            self.matrix_material == other.matrix_material,
            self.fiber_volume_fraction == other.fiber_volume_fraction
        ]
        return all(ret)

    def __str__(self):
        ret_string = f'Name: {self.name}, fiber volume fraction: {self.fiber_volume_fraction}, ' \
                     f'Fiber: {self.fiber_material.name}, Matrix: {self.matrix_material.name}'
        return ret_string

    @property
    def matrix_material(self):
        """
        return the matrix material of the fiber reinforced material

        :return: matrix material
        :rtype: ElasticMaterial or derived class
        """
        return self._matrix_material

    @property
    def fiber_material(self):
        """
        return the fiber material of the fiber reinforced material

        :return: fiber material
        :rtype: ElasticMaterial or derived class
        """
        return self._fiber_material

    @property
    def fiber_volume_fraction(self):
        """
        return the fiber volume ratio of the fiber reinforced material

        :return: fiber volume ratio
        :rtype: float
        """
        return self._fiber_volume_fraction

    def _get_e1(self):
        """
        Return the longitudinal stiffness :math:'E_1' of the fiber reinforced material

        :return: Stiffness of the fiber reinforced material in 1-direction
        :rtype: float
        """
        return self._fiber_material.get_stiffness(1) * self._fiber_volume_fraction + self._matrix_material.get_stiffness() * (1. - self._fiber_volume_fraction)

    def _get_e2(self, **kwargs):
        """
        Return the perpendicular stiffness :math:'E_2' of the fiber reinforced material

        Two methods for the calculation of :math:'E_2' are implemented, a micro mechanical and a semi-empiric
        approach, both described by Schürmann [Sch07]_. According to the Literature, the semi empiric approach gives
        results closer to experimental test results.

        :param method: (*optional*) method to use to calculate :math:`E_2`. Defaults to 'empiric'

            Options:
            - 'empiric' semi-empiric approach
            - 'default': micro-mechanical approach

        :param contraction_hindrance: set if contraction hindrance is considered in the calculation. Default is True
        :type method: str
        :type contraction_hindrance: bool

        :return: perpendicular stiffness :math:'E_2' of the composite
        :rtype: float
        """

        phi = self._fiber_volume_fraction
        eMatrix = self._matrix_material.get_stiffness()
        nuMatrix = self._matrix_material.get_poisson(12)
        e2Fiber = self._fiber_material.get_stiffness(22)
        method = kwargs.get('method', 'empiric')
        contraction_hinderance = kwargs.get('contraction_hindrance', True)

        if method == 'empiric':
            term2 = 1. + 0.85 * phi ** 2
            exponent = 1.25
        else:
            term2 = 1.
            exponent = 1.0

        if contraction_hinderance:
            term1 = eMatrix / (1. - nuMatrix ** 2)
        else:
            term1 = eMatrix

        term3 = (1. - phi) ** exponent
        term4 = term1 * phi / e2Fiber

        e2 = term1 * term2 / (term3 + term4)

        return e2

    def _get_g12(self, **kwargs):
        """
        Return the shear modulus :math:`G_12` of the composite using semi empirical relations introduced by "Förster"

        :param method: (*optional*) method to use to calculate :math:`G_12`. Default is 'foerster'

            Options:

            - 'default' miro-machanical approach
            - 'foerster': semi-empiric approach intoduced by 'Förster'

        :type method: str

        :return: shear modulus :math:`G_21` of the composite
        :rtype: float
        """

        phi = self._fiber_volume_fraction
        gMatrix = self.matrix_material.get_stiffness(12)
        g21Fiber = self.fiber_material.get_stiffness(12)
        method = kwargs.get('method', 'foerster')

        if method == 'foerster':
            term1 = 1 + 0.4 * np.power(phi, 0.5)
            term2 = np.power((1 - phi), 1.45)
            term3 = (gMatrix / g21Fiber) *phi
            g21 = gMatrix * term1 / (term2 + term3)
        else:
            g21 = gMatrix / (1. - phi + phi * gMatrix / g21Fiber)

        return g21

    def _get_nu12(self):
        """
        Return the Poisson Ratio :math:`\nu_{12}` of the fiber reinforced material using the mixing relation

        :return: Poisson Ratio :math:`\nu_{12}` of the fiber reinforced material
        :rtype: float
        """

        phi = self._fiber_volume_fraction
        nuMatrix = self._matrix_material.get_poisson(12)
        nu12Fiber = self._fiber_material.get_poisson(12)

        nu21 = (nu12Fiber * phi + nuMatrix * (1. - phi))

        return nu21

    def _get_nu23(self):
        """
        Return the Poisson Ratio :math:`\nu_{23}` of the fiber reinforced material using the relation introduced by
            'Foye'

        :return: Poisson Ratio :math:`\nu_{23}` of the fiber reinforced material
        :rtype: float
        """

        phi = self._fiber_volume_fraction
        nuMatrix = self._matrix_material.get_poisson(12)
        eMatrix = self._matrix_material.get_stiffness()
        nu32Fiber = self._fiber_material.get_poisson(32)
        nu21 = self._get_nu12()
        e1 = self._get_e1()

        term1 = 1. + nuMatrix - nu21 * eMatrix / e1
        term2 = 1. - nuMatrix ** 2 + nuMatrix * nu21 * eMatrix / e1

        nuMatrixEffektiv = nuMatrix * term1 / term2

        nu32 = phi * nu32Fiber + (1 - phi) * nuMatrixEffektiv

        return nu32


class Laminate(object):
    """
    Creates a laminate object with an empty stacking

    .. note::
        a factory class for creating laminates exists and it's use for creating laminates is encouraged
        (see :class:`StandardLaminateFactory <material_mechanics.materials.material_factories.StandardLaminateFactory>`)

    :param name: (*optional*) name of the laminate, default is None
    :param layer_stiffness_symmetry: (*optional*) sets the method of enforcement of symmetry in the layer materials
        stiffness matrix. For details on the algorithm for symmetry enforcement please see
        :func:`force_symmetry() <material_mechanics.tools.functions.force_symmetry>`. Default is 'upper'.
        Options: ('mean', 'upper', 'lower', None)
    :type name: str
    :type layer_stiffness_symmetry: str or None

    Example::

        >>> import material_mechanics as mm
        >>>
        >>> # fiber definition
        >>> name = 'Carbon Fiber HT'
        >>> stiffness = dict(e1=230000, e2=13000, g12=50000)
        >>> poisson = dict(nu12=0.23, nu23=0.3)
        >>> fiber = mm.transverse_isotropic_material(name=name, stiffness=stiffness, poisson=poisson, density=1.74)
        >>>
        >>> # matrix definition
        >>> name = 'Epoxy Resin'
        >>> matrix = mm.isotropic_material(name=name, stiffness=3200.0, poisson=0.3, density=1.2)
        >>>
        >>> # fiber volume content
        >>> phi = 0.65
        >>>
        >>> # fiber reinforced material initialization
        >>> frp_material = mm.FiberReinforcedPlastic(
        >>>     fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi
        >>> )
        >>>
        >>> lam_fzb = mm.Laminate()
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=45.))
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=90.))
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=135.))
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=0.))
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=0.))
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=135.))
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=90.))
        >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=45.))

    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name', None)
        self._layer_symmetry = kwargs.get('layer_stiffness_symmetry', 'upper')
        self.stacking = list()

    def __str__(self):
        if len(self.stacking) > 0:
            return '\n'.join([f'{i+1} - {str(layer)}' for i, layer in enumerate(self.stacking)])
        else:
            return 'This is an empty laminate'

    def add_layer(self, layer, *args, **kwargs):
        """
        add a layer to the laminate

        :param layer: layer to add to the end of the laminate
        :type layer: Layer

        :return: None

        Example::

            >>> import material_mechanics as mm
            >>>
            >>> # fiber definition
            >>> name = 'Carbon Fiber HT'
            >>> stiffness = dict(e1=230000, e2=13000, g12=50000)
            >>> poisson = dict(nu12=0.23, nu23=0.3)
            >>> fiber = mm.transverse_isotropic_material(name=name, stiffness=stiffness, poisson=poisson, density=1.74)
            >>>
            >>> # matrix definition
            >>> name = 'Epoxy Resin'
            >>> matrix = mm.isotropic_material(name=name, stiffness=3200.0, poisson=0.3, density=1.2)
            >>>
            >>> # fiber volume content
            >>> phi = 0.65
            >>>
            >>> # fiber reinforced material initialization
            >>> frp_material = mm.FiberReinforcedPlastic(
            >>>     fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi
            >>> )
            >>>
            >>> lam_fzb = mm.Laminate()
            >>> lam_fzb.add_layer(mm.Layer(name='my_layer', material=frp_material, thickness=0.25, orientation=45.))
            >>> print(lam_fzb)
            1 - Name: my_layer, Material: CarbonFiberHT_EpoxyResin_65, Thickness: 0.25, Orientation: 45.0'

        """
        self.stacking.append(layer)

    @property
    def density(self):
        r"""
        return the materials density

        :return: density in :math:`\frac{g}{cm^3}`
        :rtype: float
        """
        cummulative_density = 0.0
        for layer_ in self.stacking:
            cummulative_density += layer_.material.density * layer_.thickness

        return cummulative_density / self.thickness

    @property
    def thickness(self):
        r"""
        Return the laminates thickness

        :return: laminate thickness in :math:`mm`
        :rtype: float
        """
        return sum([l.thickness for l in self.stacking])

    @property
    def area_weight(self):
        r"""
        return the laminates area weight

        :return: laminate area weight in :math:`\frac{g}{cm^2}`
        :rtype: float
        """
        return sum([l.get_area_weight for l in self.stacking])

    @property
    def z_values(self):
        """
        return the lower and top coordinate of each layer

        :return: List of tupels. Each tuple holds the upper an lower coordinate of the layer in reference to the
            laminate central plane
        :rtype: list of tuples
        """
        z_values = list()
        for layer in self.stacking:
            if len(z_values) == 0:
                z_lower = -self.thickness * 0.5
                z_upper = z_lower + layer.thickness
            else:
                z_lower = z_upper
                z_upper = z_lower + layer.thickness
            z_values.append((z_lower, z_upper))
        return z_values

    @property
    def abd_matrix(self):
        """
        return the abd Matrix of the Laminate using classical lamination theory

        :return: stiffness matrix of the combined shell and plate element
        :rtype: numpy.ndarray
        """

        abd = np.concatenate(
            (np.concatenate((self.a_matrix, self.b_matrix), axis=1),
             np.concatenate((self.b_matrix, self.d_matrix), axis=1)), axis=0)

        return abd

    @property
    def a_matrix(self):
        """
        return the shell stiffness matrix of the laminate as defined in the classical lamination theory

        :return: shell stiffness matrix
        :rtype: numpy.ndarray
        """

        matrices = [layer.stiffness_matrix(symmetry=self._layer_symmetry) * layer.thickness for layer in self.stacking]
        return np.sum(matrices, axis=0)

    @property
    def b_matrix(self):
        """
        Return the coupling matrix of the laminate as defined in the classical lamination theory

        :return: coupling matrix
        :rtype: numpy.ndarray
        """

        matrices = [layer.stiffness_matrix(symmetry=self._layer_symmetry) * (zVals[1] ** 2 - zVals[0] ** 2) for layer, zVals in zip(self.stacking, self.z_values)]
        return 0.5 * np.sum(matrices, axis=0)

    @property
    def d_matrix(self):
        """
        calculates the plate stiffness matrix of the laminate as defined in the classical lamination theory

        :return: plate stiffness matrix
        :rtype: numpy.ndarray
        """

        matrices = [layer.stiffness_matrix(symmetry=self._layer_symmetry) * (zVals[1] ** 3 - zVals[0] ** 3) for layer, zVals in zip(self.stacking, self.z_values)]
        return np.sum(matrices, axis=0) / 3.

    @property
    def layer_count(self):
        """
        Return the number of layers in the laminate

        :return: number of layers
        :rtype: int
        """
        return len(self.stacking)

    def get_strains(self, line_loads):
        r"""
        return the global laminate strains resulting from a planar external loading.

        The laminate strains are calculated through the laminates :math:`ABD` matrix. The external load  is defined by
        the line load vector :math:`[n_x, n_y, n_{xy}, m_x, m_y, m_{xy}]`. The line loads :math:`n` are given in
        :math:`\frac{N}{mm}` and the line moments :math:`m` in :math:`N`

        :param line_loads: line load vector
        :type line_loads: numpy.ndarray
        :return: laminate strains
        :rtype: numpy.ndarray

        Example::

            >>> import material_mechanics as mm
            >>>
            >>> # fiber definition
            >>> name = 'Carbon Fiber HT'
            >>> stiffness = dict(e1=230000, e2=13000, g12=50000)
            >>> poisson = dict(nu12=0.23, nu23=0.3)
            >>> fiber = mm.transverse_isotropic_material(name=name, stiffness=stiffness, poisson=poisson, density=1.74)
            >>>
            >>> # matrix definition
            >>> name = 'Epoxy Resin'
            >>> matrix = mm.isotropic_material(name=name, stiffness=3200.0, poisson=0.3, density=1.2)
            >>>
            >>> # fiber volume content
            >>> phi = 0.65
            >>>
            >>> # fiber reinforced material initialization
            >>> frp_material = mm.FiberReinforcedPlastic(
            >>>     fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi
            >>> )
            >>>
            >>> lam_fzb = mm.Laminate()
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=45.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=90.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=135.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=0.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=0.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=135.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=90.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=45.))
            >>>
            >>> line_loads = np.array([100, 50, 10, 20, 5, 3])
            >>>
            >>> lam_fzb.get_strains(line_loads=line_loads)
            array([ 0.00071862,  0.00017637,  0.0002169 ,  0.00100021, -0.00015305, -0.0003249 ])

        """
        return np.dot(np.linalg.inv(self.abd_matrix), line_loads)

    def get_layer_strains(self, laminate_strains):
        r"""
        return the strains in the layers of the laminate resulting from an external planar loading of the laminate

        The laminate strains are :math:`[\varepsilon_x, \varepsilon_y, \gamma_{xy}, \kappa_x, \kappa_y, \kappa_{xy}]`
        and are the resulting global laminate strains from an external loading of the laminate, calculated by the
        classical laminate theory (CLT). This step is done in the
        :func:`get_strains() <material_mechanics.materials.composites.Laminate.get_strains>`

        :param laminate_strains: global laminate strains
        :return: strains of the layers at the bottom and the top of each layer
        :rtype: list of lists of numpy.ndarray

        Example::

            >>> import material_mechanics as mm
            >>>
            >>> # fiber definition
            >>> name = 'Carbon Fiber HT'
            >>> stiffness = dict(e1=230000, e2=13000, g12=50000)
            >>> poisson = dict(nu12=0.23, nu23=0.3)
            >>> fiber = mm.transverse_isotropic_material(name=name, stiffness=stiffness, poisson=poisson, density=1.74)
            >>>
            >>> # matrix definition
            >>> name = 'Epoxy Resin'
            >>> matrix = mm.isotropic_material(name=name, stiffness=3200.0, poisson=0.3, density=1.2)
            >>>
            >>> # fiber volume content
            >>> phi = 0.65
            >>>
            >>> # fiber reinforced material initialization
            >>> frp_material = mm.FiberReinforcedPlastic(
            >>>     fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi
            >>> )
            >>>
            >>> lam_fzb = mm.Laminate()
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=45.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=90.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=135.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=0.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=0.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=135.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=90.))
            >>> lam_fzb.add_layer(mm.Layer(material=frp_material, thickness=0.25, orientation=45.))
            >>>
            >>> line_loads = np.array([100, 50, 10, 20, 5, 3])
            >>>
            >>> laminate_strains = lam_fzb.get_strains(line_loads=line_loads)
            >>> layer_strains = fzb_lam.get_layer_strains(laminate_strains=laminate_strains)
            >>>
            >>> for i, ls in enumerate(layer_strains):
            >>>     print(f'Layer {i+1}\nBottom strain:{ls[0]}, Top strain:{ls[1]}')
            Layer 1
            Bottom strain:[ 0.00029482 -0.00024698  0.000611  ], Top strain:[ 0.0003601  -0.00010048  0.00032269]
            Layer 2
            Bottom strain:[ 2.91153430e-04 -3.15320675e-05 -4.60576048e-04], Top strain:[ 0.00025289  0.00021852 -0.00037935]
            Layer 3
            Bottom strain:[ 4.60294158e-05  4.25381385e-04 -3.43704880e-05], Top strain:[0.00019254 0.00049066 0.00025394]
            Layer 4
            Bottom strain:[0.00046857 0.00021463 0.00029813], Top strain:[0.00071862 0.00017637 0.0002169 ]
            Layer 5
            Bottom strain:[0.00071862 0.00017637 0.0002169 ], Top strain:[0.00096868 0.0001381  0.00013568]
            Layer 6
            Bottom strain:[0.00048555 0.00062123 0.00083057], Top strain:[0.00063206 0.00068651 0.00111889]
            Layer 7
            Bottom strain:[ 9.98395044e-05  1.21872905e-03 -5.44556546e-05], Top strain:[6.15767194e-05 1.46878128e-03 2.67684241e-05]
            Layer 8
            Bottom strain:[ 0.00075179  0.00077856 -0.0014072 ], Top strain:[ 0.00081708  0.00092507 -0.00169552]
        """
        layer_vals = list()
        for l, z_vals in zip(self.stacking, self.z_values):
            layer_val = list()
            for z in z_vals:
                layer_val.append(np.dot(get_T_strain_2d(np.pi * l.orientation / 180.), get_strain_at_z(laminate_strains, z)))
            layer_vals.append(tuple(layer_val))
        return layer_vals


class Layer(object):
    """
    Layer class to be used in the Laminate class
    
    :param material: The material of the layer
    :param thickness: thickness of the layer in :math:`mm`
    :param orientation: orientation of the layer in :math:`degrees`
    :param name: (*optional*) name to reference the layer by
    :type material: ElasticMaterial or derived class
    :type thickness: float
    :type orientation: float
    :type name: str or None

    Example::

        >>> import material_mechanics as mm
        >>>
        >>> # fiber definition
        >>> name = 'Carbon Fiber HT'
        >>> stiffness = dict(e1=230000, e2=13000, g12=50000)
        >>> poisson = dict(nu12=0.23, nu23=0.3)
        >>> fiber = mm.transverse_isotropic_material(name=name, stiffness=stiffness, poisson=poisson, density=1.74)
        >>>
        >>> # matrix definition
        >>> name = 'Epoxy Resin'
        >>> matrix = mm.isotropic_material(name=name, stiffness=3200.0, poisson=0.3, density=1.2)
        >>>
        >>> # fiber volume content
        >>> phi = 0.65
        >>>
        >>> # fiber reinforced material initialization
        >>> frp_material = mm.FiberReinforcedPlastic(
        >>>     fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi
        >>> )
        >>> layer = mm.Layer(name='my_layer',material=frp_material, thickness=0.25, orientation=45.)
        >>> print(layer)
        Name: my_layer, Material: CarbonFiberHT_EpoxyResin_65, Thickness: 0.25, Orientation: 45.0

    """

    def __init__(self, material, thickness, orientation, *args, **kwargs):
        self._material = material
        self._thickness = thickness
        self._orientation = orientation
        self._ply_name = kwargs.get('name', None)

    def __str__(self):
        ret_str = f'Name: {self._ply_name}, Material: {self._material.name}, Thickness: {self._thickness}, ' \
                  f'Orientation: {self._orientation}'
        return ret_str

    @property
    def material(self):
        """
        return layer material

        :return: layer material
        :rtype: ElasticMaterial or derived class
        """
        return self._material

    @property
    def thickness(self):
        """
        return layer thickness

        :return: layer thickness in :math:`mm`
        :rtype: float
        """
        return self._thickness

    @property
    def orientation(self):
        """
        return layer orientation

        :return: layer orientation in :math:`degree`
        :rtype: float
        """
        return self._orientation

    @property
    def name(self):
        """
        return the name of the layer

        :return: layer name
        :rtype: str
        """
        return self._ply_name

    @name.setter
    def name(self, name):
        """
        Set the name of the layer
        :param name: new layer name
        :type name: str
        """
        assert isinstance(name, str)
        self._ply_name = name

    @property
    def area_weight(self):
        r"""
        return the area weight

        :return: area weight of the layer in :math:`\frac{g}{cm^2}`
        :rtype: float
        """
        return self._material.density * self._thickness * 0.1

    def compliance_matrix(self, *args, **kwargs):
        """
        calculate the compliance matrix of the layer in the laminate coordinate system (i.e. in the 0° direction)

        :return: layer compliance matrix
        :rtype: numpy.ndarray
        """
        symmetry = kwargs.get('symmetry', None)

        c = np.cos(self._orientation * np.pi / 180.)
        s = np.sin(self._orientation * np.pi / 180.)
        c2 = np.cos(2. * self._orientation * np.pi / 180.)
        s2 = np.sin(2. * self._orientation * np.pi / 180.)

        t_sigma_xy12 = np.array([
            [c ** 2, s ** 2, s2],
            [s ** 2, c ** 2, -s2],
            [-.5 * s2, .5 * s2, c2]])

        t_epsilon_12xy = np.array([
            [c ** 2, s ** 2, -.5 * s2],
            [s ** 2, c ** 2, .5 * s2],
            [s2, -s2, c2]])

        material_compliance_matrix = force_symmetry(self._material.compliance_matrix_2d, symmetry=symmetry)

        return np.matmul(t_epsilon_12xy, np.matmul(material_compliance_matrix, t_sigma_xy12))

    def stiffness_matrix(self, *args, **kwargs):
        """
        calculates the stiffness matrix of the layer in the laminate coordinate system
        
        :return: layer stiffness matrix
        :rtype: numpy.ndarray
        """
        symmetry = kwargs.get('symmetry', None)

        return np.linalg.inv(self.compliance_matrix(symmetry=symmetry))

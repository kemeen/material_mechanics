#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import abc
import logging

from .elastic_materials import ElasticMaterial
from .composites import Laminate, FiberReinforcedPlastic, Layer

# ====================
# Set up module logger
# ====================
logger = logging.getLogger(__name__)


class LayerFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_layer(self, *args, **kwargs): pass


class LaminateFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_laminate(self, *args, **kwargs): pass


class FrpFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_material(self, *args, **kwargs): pass


class MaterialFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_material(self, stiffness, poisson, strength, density, *args, **kwargs): pass


class PuckSetFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_material(self, *args, **kwargs): pass


# Todo add Examples for each class/method

# ===============
# Layer Factories
# ===============
class StandardLayerFactory(LayerFactory):
    """
    Standard Factory for Layers

    Standard values for the parameters may be provided at initiation. Parameters that did not receive a value at
    initiation, need to be set when the get_layer method is called. If not a ValueError is raised.

    :param material: Layer material
    :param thickness: Thickness of the layer in :math:`mm`
    :param orientation: Orientation of the Layerin :math:`degree`
    :type material: ElasticMaterial or None
    :type thickness: float or None
    :type orientation: float or None

    .. note::

        All materials derived from
        :class:`ElasticMaterial <material_mechanics.materials.elastic_materials.ElasticMaterial>`, like
        :class:`FiberReinforcedPlastic <material_mechanics.materials.composites.FiberReinforcedPlastic>`,
        can be used in a Layer.

    Example::

        import material_mechanics as mm

        stiffness = dict(e1=140000, e2=9000, g12=4600)
        poisson = dict(nu12=0.3, nu23=0.37)
        density = 1.5

        cfk = mm.transverse_isotropic_material(
            name='CFK',
            stiffness=stiffness,
            poisson=poisson,
            strength=None,
            density=density)

        steel = mm.isotropic_material(
            name='Steel',
            stiffness=200000.0,
            poisson=0.34,
            strength=700,
            density=7.9
        )

        FixedMaterialLayerCreator = mm.StandardLayerFactory(material=cfk)
        layer_1 = FixedMaterialLayerCreator.get_layer(thickness=0.1, orientation=30)
        layer_2 = FixedMaterialLayerCreator.get_layer(thickness=0.1, orientation=60)
        layer_3 = FixedMaterialLayerCreator.get_layer(thickness=0.1, orientation=90)


        FixedThicknessLayerCreator = mm.StandardLayerFactory(thickness=0.2)
        layer_1 = FixedThicknessLayerCreator.get_layer(material=steel, orientation=30)
        layer_2 = FixedThicknessLayerCreator.get_layer(material=cfk, orientation=45)
        layer_3 = FixedThicknessLayerCreator.get_layer(material=steel, orientation=60)

        # standard values may also be overwritten
        layer_3 = FixedThicknessLayerCreator.get_layer(thickness=0.1, material=steel, orientation=60)

    """

    def __init__(self, *args, **kwargs):
        self._material = kwargs.get('material', None)
        self._thickness = kwargs.get('thickness', None)
        self._orientation = kwargs.get('orientation', None)

    def get_layer(self, *args, **kwargs):
        """
        Return a layer with a material, thickness and orientation.

        If the values for material, thickness and/or orientation where provided at initiation,
        they may be omitted or be overwritten.

        :param material: (*optional*) Layer material, default is material provided at initialization of class
        :param thickness: (*optional*) Thickness of the layer, default is thickness provided at initialization of class
        :param orientation: (*optional*) Orientation of the Layer, default is orientation provided at initialization of class
        :type material: ElasticMaterial
        :type thickness: float
        :type orientation: float
        :return: Layer
        :rtype: Layer
        """
        material = kwargs.get('material', self._material)
        thickness = kwargs.get('thickness', self._thickness)
        orientation = kwargs.get('orientation', self._orientation)

        if any([item is None for item in (material, thickness, orientation)]):
            raise ValueError('Please provide a material, thickness and orientation for the layer')

        return Layer(material=material, thickness=thickness, orientation=orientation)


# ==================
# Laminate Factories
# ==================
class StandardLaminateFactory(LaminateFactory):
    """
    Factory class for creating laminates

    :param material:
        (*optional*) Default material for each layer of the laminate.
        May be overwritten in layer dicts. Default is None

    :param layer_thickness:
        (*optional*) Default layer thickness for the laminate.
        May be overwritten in layer dicts. Default is None

    :param symmetry:
        (*optional*) symmetry mode of laminate, default is None

        Options:

        - None: No symmetry is forced
        - 'center_layer': The symmetry plane of the last layer is the symmetry plane for the laminate, so all but the last layer are mirrored by this plane
        - 'full': (*catches all strings but 'center_layer'*) all layers are mirrored at the plane defining the lower border of the last plane, including the last layer

    :type material: ElasticMaterial or derived
    :type layer_thickness: float or None
    :type symmetry: str or None
    """

    def __init__(self, *args, **kwargs):
        self._material = kwargs.get('material', None)
        self._layer_thickness = kwargs.get('layer_thickness', None)
        self._symmetry = kwargs.get('symmetry', None)
        self._layer_factory = StandardLayerFactory(material=self._material, thickness=self._layer_thickness)

    def get_laminate(self, layers, *args, **kwargs):
        r"""
        Return a Laminate

        :param layers: stacking of a laminate defined as dicts. each dict holds the parameters of the layer
            ('thickness' in :math:`mm`, 'orientation' in :math:`degree`, 'material'). 'thickness' and 'material' may be
            set at the initiation of the class, in which case they may be omitted in the layer dicts. If provided at
            method call, init values of parameters will be overwritten.
        :param symmetry: (*optional*) symmetry mode of laminate, default is None

            Options:

            - None: No symmetry is forced
            - 'center_layer': The symmetry plane of the last layer is the symmetry plane for the laminate, so all but the last layer are mirrored by this plane
            - 'full': (*catches all strings but 'center_layer'*) all layers are mirrored at the plane defining the lower border of the last plane, including the last layer

        :type layers: list of dicts
        :type symmetry: str or None
        :return: Laminate
        :rtype: Laminate
        """
        symmetry = kwargs.get('symmetry', None)
        laminate = Laminate()

        for l in layers:
            layer = self._layer_factory.get_layer(**l)
            laminate.add_layer(layer=layer)

        if symmetry is not None:
            # define symmetry layers according to provided symmetry switch
            sym_layers = list(reversed(layers))
            if symmetry == 'center_layer':
                sym_layers.pop()

            # add layers to achieve symmetry
            for layer in sym_layers:
                laminate.add_layer(layer=layer)

        return laminate


# class SymmetricalLaminateFactory(LaminateFactory):
#     """
#     Returns a symmetrical laminate
#     """
#
#     def get_laminate(self, layers, *args, **kwargs):
#         """
#         Returns a symmetric laminate
#
#         Achieves symmetry by adding the provided layers in reverse order to the layer stack.
#         The middle layer can be left out when adding the new layers to move the plain of symmetry into the middle layer.
#
#         :param layers: List of Layer objects
#         :param symmetry: flag to define position of symmetry plane
#
#             Options:
#
#                 - 0: between the two central layers (default)
#                 - 1: in the central layer (central layer is not mirrored)
#
#         :type layers: list of Layers
#         :type symmetry: (*optional*) int or None
#         :return: symmetrical laminate
#         :rtype: Laminate
#         """
#         symmetry = kwargs.get('symmetry', 0)
#         laminate = Laminate()
#
#         for layer in layers:
#             laminate.add_layer(layer=layer)
#
#         # define symmetry layers according to provided symmetry switch
#         sym_layers = list(reversed(layers))
#         if symmetry == 1:
#             sym_layers.pop()
#
#         # add layers to achieve symmetry
#         for layer in sym_layers:
#             laminate.add_layer(layer=layer)
#
#         return laminate
#
#
# class SingleMaterialLaminateFactory(LaminateFactory):
#     """
#     Factory class for creating laminates where each layer has the same material
#
#     :param material: The material for each layer of the laminate
#     :type material: ElasticMaterial
#
#     """
#
#     def __init__(self, material, *args, **kwargs):
#         self.material = material
#         self._layer_factory = StandardLayerFactory(material=material)
#
#     def get_laminate(self, layers, *args, **kwargs):
#         r"""
#         Returns a Laminate where each layer consists of the same material
#
#         :param layers: stacking of laminate defined as pairs of thickness in :math:`mm` and orientation in :math:`°`
#         :param symmetry: (*optional*) symmetry mode of laminate, default is None
#
#             options
#
#                 - None: No symmetry is forced
#                 - 'center_layer': The symmetry plane of the last layer is the symmetry plane for the laminate, so all but the last layer are mirrored by this plane
#                 - 'full': (*catches all strings but 'center_layer'*) all layers are mirrored at the plane defining the lower border of the last plane, including the last layer
#
#         :return: a Laminate with each layer having the same base material
#         :rtype: Laminate
#         """
#         symmetry = kwargs.get('symmetry', None)
#         laminate = Laminate()
#
#         for t, ori in layers:
#             layer = self._layer_factory.get_layer(material=self.material, thickness=t, orientation=ori)
#             laminate.add_layer(layer=layer)
#
#         if symmetry is not None:
#             # define symmetry layers according to provided symmetry switch
#             sym_layers = list(reversed(layers))
#             if symmetry == 'center_layer':
#                 sym_layers.pop()
#
#             # add layers to achieve symmetry
#             for layer in sym_layers:
#                 laminate.add_layer(layer=layer)
#
#         return laminate


# ==================================
# Fiber reinforced plastic factories
# ==================================
class ChangedFvcFrp(FrpFactory):
    """
    A factory class to get an FRP material from an existing FRP with changed fiber volume content

    :param material: the base FRP Material from which to calculate the generated materials
    :type material: FiberReinforcedPlastic
    """

    def __init__(self, material, *args, **kwargs):
        self._material = material

    def get_material(self, phi, *args, **kwargs):
        """
        return a fiber reinforced plastic with the same constituents as the base fiber reinforced plastic but changed
        fiber volume content

        :param phi: fiber volume content of the new fiber reinforced plastic
        :type phi: float
        :return: a fiber reinforced plastic material with the provided fiber volume content
        :rtype: FiberReinforcedPlastic
        """

        if phi is None or self._material.fiber_volume_fraction == phi:
            return self._material

        material_name = '{fiber}_{matrix}_{fvc}'.format(
            fiber=self._material.fiber_material.name,
            matrix=self._material.matrix_material.name,
            fvc=int(phi * 100)
        ).replace(" ", "")

        material = FiberReinforcedPlastic(
            name=material_name,
            fiber_material=self._material.fiber_material,
            matrix_material=self._material.matrix_material,
            fiber_volume_fraction=phi
        )

        return material


# =========
# Puck Sets
# =========
class ChangedFvcPuckSet(PuckSetFactory):
    """
    A factory class to generate puck sets from an existing puck set with changed fiber volume content

    :param puck_set: the base puck set from which to generate new puck sets
    :type puck_set: PuckStrengthSet
    """

    def __init__(self, puck_set, resin_strength):
        self._puck_set = puck_set
        self._resin_strength = resin_strength

    def get_material(self, phi, *args, **kwargs):
        """
        returns a puck set with changed fiber volume content based on the puck set provided at initiation

        :param phi: fiber volume content of the new puck set
        :type phi: float
        :return: puck set with the provided fiber volume content
        :rtype: PuckStrengthSet
        """

        if phi is None or phi == self._puck_set.phi:
            return self._puck_set

        puck_set = copy.deepcopy(self._puck_set)
        puck_set.r_lp = (self._puck_set.r_lp - self._resin_strength) * phi / self._puck_set.phi + self._resin_strength
        puck_set.r_lm = (self._puck_set.r_lm - self._resin_strength) * phi / self._puck_set.phi + self._resin_strength
        puck_set.phi = phi

        return puck_set


# =========
# Materials
# =========
def isotropic_material(stiffness, poisson, density, *args, **kwargs):
    r"""
    create a material with isotropic properties

    :param density: material density in :math:`\frac{g}{cm^3}`
    :type density: float
    :param stiffness: material stiffness in :math:`\frac{N}{mm^2}`
    :type stiffness: float
    :param poisson: material poisson's ratio
    :type poisson: float
    :param strength: material strength in :math:`\frac{N}{mm^2}`
    :type strength: float

    :return: material with isotropic properties
    :rtype: ElasticMaterial
    """
    name = kwargs.get('name', None)
    strength = kwargs.get('strength', None)

    g = stiffness / (2 * (1 + poisson))
    stiffness = [(stiffness, g)] * 3

    poisson = [(poisson, poisson)] * 3

    if strength is not None:
        strength = [strength] * 3

    material = ElasticMaterial(
        name=name,
        stiffness=stiffness,
        poisson=poisson,
        strength=strength,
        density=density
    )

    return material


def transverse_isotropic_material(stiffness, poisson, density, *args, **kwargs):
    r"""
    create a material with transverse isotropic properties

    .. note::
        poisson ratios are in international notation.
        The first index points to the causing strain.
        The second index points to the resulting strain.

    :param density: material density in :math:`\frac{g}{cm^3}`
    :type density: float
    :param stiffness: dictionary of material stiffnesses in :math:`\frac{N}{mm^2}`
    :type stiffness: dict
    :param poisson: material poisson ratios
    :type poisson: dict
    :param strength: dictionary of material strengths in :math:`\frac{N}{mm^2}`
    :type strength: dict
    :return: material with isotropic properties
    :rtype: ElasticMaterial
    """

    name = kwargs.get('name', None)
    strength = kwargs.get('strength', None)

    e1 = float(stiffness['e1'])
    e2 = float(stiffness['e2'])
    g12 = float(stiffness['g12'])

    nu12 = float(poisson['nu12'])
    nu23 = float(poisson['nu23'])
    nu21 = e2 / e1 * nu12

    g23 = e2 / (2 * (1 + nu23))

    stiffnesses = [(e1, g23), (e2, g12), (e2, g12)]
    poissons = [(nu23, nu23), (nu12, nu21), (nu12, nu21)]

    # TODO: implement strength calculation from input dictionary
    # Input type for strength is {'r1_plus': xxx, 'r1_minus': xxx, 'r2_plus': xxx, 'r2_minus': xxx, r12: xxx}

    material = ElasticMaterial(
        name=name,
        stiffness=stiffnesses,
        poisson=poissons,
        strength=strength,
        density=density)

    return material


def orthotropic_material(stiffness, poisson, density, *args, **kwargs):
    r"""
    create a material with orthotropic properties

    .. note::
        poisson ratios are in international notation.
        The first index points to the causing strain.
        The second index points to the resulting strain.

    :param density: material density in :math:`\frac{g}{cm^3}`
    :type density: float
    :param stiffness: dictionary of material stiffness in :math:`\frac{N}{mm^2}`
    :type stiffness: dict
    :param poisson: material poisson ratios
    :type poisson: dict
    :param strength: dictionary of material strengths in :math:`\frac{N}{mm^2}`
    :type strength: dict
    :return: material with isotropic properties
    :rtype: ElasticMaterial
    """

    name = kwargs.get('name', None)
    strength = kwargs.get('strength', None)

    e1 = float(stiffness['e1'])
    e2 = float(stiffness['e2'])
    e3 = float(stiffness['e3'])
    g12 = float(stiffness['g12'])
    g13 = float(stiffness['g13'])
    g23 = float(stiffness['g23'])

    nu12 = float(poisson['nu12'])
    nu21 = float(poisson['nu21'])
    nu13 = float(poisson['nu13'])
    nu31 = float(poisson['nu31'])
    nu23 = float(poisson['nu23'])
    nu32 = float(poisson['nu32'])

    # TODO: implement strength calculation from input dictionary
    # Input type for strength is {'r1_plus': xxx, 'r1_minus': xxx, 'r2_plus': xxx, 'r2_minus': xxx, r12: xxx}

    material = ElasticMaterial(
        name=name,
        stiffness=[(e1, g23), (e2, g13), (e3, g12)],
        poisson=[(nu23, nu32), (nu13, nu31), (nu12, nu21)],
        strength=strength,
        density=density)

    return material

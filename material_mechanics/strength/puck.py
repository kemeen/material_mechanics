"""
This module provides all necessary tools to calculate the two- and three dimensional puck material exertions of fiber
reinforced plastics. For further reading and details on the theory please check out the refereed literature.

.. rubric:: Literature

.. [Sch07] *German* H. Schürmann, Konstruieren mit Faser-Kunststoff-Verbunden. Berlin, Heidelberg, 2007.
.. [Puc04] *English* A. Puck and H. Schürmann, “Failure analysis of FRP laminates by means of physically based
    phenomenological models,” Fail. Criteria Fibre-Reinforced-Polymer Compos., pp. 832–876, Jan. 2004.
.. [Puc02] *English* [1] A. Puck, J. Kopp, and M. Knops, “Guidelines for the determination of the parameters in Puck’s
    action plane strength criterion,” Compos. Sci. Technol., vol. 62, no. 3, pp. 371–378, Feb. 2002.
"""
import logging
import numpy as np

from .. import FiberReinforcedPlastic

# ====================
# Set up module logger
# ====================
logger = logging.getLogger(__name__)


class PuckStrengthSet(object):
    r"""
    Class providing the necessary methods to calculate the Puck Criterion

    For further details please check [Sch07]_ [Puc04]_ [Puc02]_

    :param material: composite material for the puck strength criterion
    :param p_plp: (*optional*) slope parameter  :math:`p_{\perp\parallel}^+`, default: 0.27
    :param p_plm: (*optional*) slope parameter  :math:`p_{\perp\parallel}^-`, default: 0.27
    :param p_ppp: (*optional*) slope parameter  :math:`p_{\perp\perp}^+`, default: 0.3
    :param p_ppm: (*optional*) slope parameter  :math:`p_{\perp\perp}^-`, default: 0.35
    :param m_sf: (*optional*) magnification factor :math:`m_{\sigma,f}` for fiber strain :math:`\varepsilon_{1}` due to
        perpendicular stress (:math:`\sigma_{2}`, :math:`\sigma_{3}`), default: 1.1
    :type material: FiberReinforcedPlastic
    :type p_plp: float
    :type p_plm: float
    :type p_ppp: float
    :type p_ppm: float
    :type m_sf: float
    """

    def __init__(self, material, *args, **kwargs):
        assert isinstance(material, FiberReinforcedPlastic)

        # material properties
        self.nu_f_ql = material.fiber_material.get_poisson(index=12)
        self.e_l = material.get_stiffness()
        try:
            self.e_f_l = material.fiber_material.get_stiffness()
        except AttributeError:
            self.e_f_l = kwargs.get('e_fiber', None)
        self.nu_ql = material.get_poisson(index=12)
        try:
            self._phi = material.fiber_volume_fraction
        except AttributeError:
            self._phi = kwargs.get('phi', None)

        # curve parameters
        self.p_ppm = kwargs.get('p_ppm', 0.27)
        self.p_ppp = kwargs.get('p_ppp', 0.27)
        self.p_plm = kwargs.get('p_plm', 0.3)
        self.p_plp = kwargs.get('p_plp', 0.35)
        self.m_sf = kwargs.get('m_qf', 1.1)

        # Strength properties
        self.r_lp = material.get_strength(index=11, direction='+')
        self.r_lm = material.get_strength(index=11, direction='-')
        self.r_pp = material.get_strength(index=22, direction='+')
        self.r_pm = material.get_strength(index=11, direction='-')
        self.r_pl = material.get_strength(index=12)
        self.r_ppa = self.r_pm / (2 * (1 + self.p_ppm))

    def inter_fiber_exertion(self, stress_vector, theta):
        r"""
        calculates the inter-fiber exertion in the plane defined by the angle :math:`\Theta`

        :param stress_vector: stress in fiber coordinates (:math:`\sigma_{11}`, :math:`\sigma_{22}`,
            :math:`\sigma_{33}`, :math:`\sigma_{23}`, :math:`\sigma_{31}`, :math:`\sigma_{21}`)
        :param theta: angle defining the plane in radians
        :type stress_vector: numpy.ndarray
        :type theta: float
        :return: inter fiber exertion
        """

        relevant_stresses = stress_vector[1:]
        if all(relevant_stresses == 0):
            return 0

        sig_n, tau_nt, tau_n1 = np.dot(get_stress_transformation_matrix(theta), relevant_stresses)

        if tau_nt == 0:
            c = 0
        else:
            c = tau_nt ** 2 / (tau_nt ** 2 + tau_n1 ** 2)

        if tau_n1 == 0:
            s = 0
        else:
            s = tau_n1 ** 2 / (tau_nt ** 2 + tau_n1 ** 2)

        if sig_n >= 0:
            form_factor = self.p_ppp / self.r_ppa * c + self.p_plp / self.r_pl * s
            exertion = np.sqrt(((1. / self.r_pp - form_factor) * sig_n) ** 2 + (tau_nt / self.r_ppa) ** 2 + (
                tau_n1 / self.r_pl) ** 2) + form_factor * sig_n
        else:
            form_factor = self.p_ppm / self.r_ppa * c + self.p_plm / self.r_pl * s
            exertion = np.sqrt((tau_nt / self.r_ppa) ** 2 + (tau_n1 / self.r_pl) ** 2 + (
                form_factor * sig_n) ** 2) + form_factor * sig_n

        return exertion

    def get_max_inter_fiber_exertion(self, stress_vector, *args, **kwargs):
        r"""
        returns the maximum inter fiber strain and the angle of the plane in which it occurs

        :param stress_vector: stress vector in th fiber coordinate system (:math:`\sigma_{11}`, :math:`\sigma_{22}`,
            :math:`\sigma_{33}`, :math:`\sigma_{23}`, :math:`\sigma_{31}`, :math:`\sigma_{21}`)
        :param precision: (*optional*) number of decimal points to which to return the calculated inter-fiber exertion,
            default is 2
        :param angle_precision: (*optional*) number of decimal points to which to calculate the angle of the plane in
            which maximum strain occurs, default is 2
        :type stress_vector: numpy.ndarray
        :type precision: int
        :type angle_precision: int
        :return: 2-tuple

            - maximum inter fiber strain
            - angle of plane in which maximum strain occurs
        """
        assert isinstance(stress_vector, np.ndarray)

        precision = kwargs.get('precision', 2)
        angle_precision = kwargs.get('angle_precision', 2)

        relevant_stresses = stress_vector[1:]
        if all(relevant_stresses == 0):
            return 0

        offset = 0.0
        func = lambda x: 1. / (offset + self.inter_fiber_exertion(stress_vector, x * np.pi / 180.))
        inter_fiber_exertion, fracture_angle = find_min_stress_angle(func, offset=offset, precision=angle_precision)
        return round(inter_fiber_exertion, precision), fracture_angle

    def get_fiber_exertion(self, stress_vector, *args, **kwargs):
        r"""
        returns fiber strain for provided stress vector

        :param stress_vector: stress vector in fiber coordinate system (:math:`\sigma_{11}`, :math:`\sigma_{22}`,
            :math:`\sigma_{33}`, :math:`\sigma_{23}`, :math:`\sigma_{31}`, :math:`\sigma_{21}`)
        :param precision: (*optional*) number of decimal points to which to return the calculated fiber exertion,
            default is 2
        :type stress_vector: numpy.ndarray
        :type precision: int
        :return: fiber strain
        """
        assert isinstance(stress_vector, np.ndarray)

        precision = kwargs.get('precision', 2)

        if all(stress_vector[:3] == 0):
            return 0.
        sigma1 = stress_vector[0]
        sigma2 = stress_vector[1]
        sigma3 = stress_vector[2]

        if sigma1 >= 0:
            exertion = np.abs(1. / self.r_lp * (sigma1 - (self.nu_ql - self.nu_f_ql * self.e_l / self.e_f_l * self.m_sf) * (sigma2 + sigma3)))
        else:
            exertion = np.abs(1. / self.r_lm * (sigma1 - (self.nu_ql - self.nu_f_ql * self.e_l / self.e_f_l * self.m_sf) * (sigma2 + sigma3)))

        return round(exertion, precision)

    def puck_exertion_2d(self, stress_vector, *args, **kwargs):
        r"""
        calculates the puck exertions under the assumption of a two-dimensional stress state

        For details on the meaning of the parameters s and m, please check [Sch07]_ [Puc04]_ [Puc02]_

        :param stress_vector: stress vector in fiber coordinate system (:math:`\sigma_{11}`, :math:`\sigma_{22}`,
            :math:`\sigma_{33}`, :math:`\sigma_{23}`, :math:`\sigma_{31}`, :math:`\sigma_{21}`)
        :param s: (*optional*) fraction of fiber exertion at which :math:`\sigma_{11}` starts to influence damage initiation
        :param m: (*optional*) minimal value of :math:`\eta_w` at which damage due to fiber and inter-fiber exertion simultaniously
            occur.
        :param ret_type: (*optional*) sets the return format for the results ('tuple', 'array', 'dict'), default is
            'tuple'
        :param precision: (*optional*) number of decimal points to which to return the calculated exertions,
            default is 2
        :type stress_vector: numpy.ndarray
        :type s: float
        :type m: float
        :type ret_type: str
        :type precision: int
        :return: fiber exertion (*fe_fb*), inter-fiber exertion without :math:`\sigma_{11}` influence (*fe_0*),
            inter-fiber exertion including :math:`\sigma_{11}` influence (*fe_1*), fracture mode (*mode*)

            formats:

            - 'tuple' (*default*): tuple(fe_fb, fe_0, fe_1, mode)
            - 'array': tuple(np.array([fe_fb, fe_0, fe_1]), mode)
            - 'dict': dict(fe_fb=fe_fb, fe_0=fe_0, fe_1=fe_1, mode=mode)
        """
        s = kwargs.get('s', 0.5)
        m = kwargs.get('m', 0.5)
        ret_type = kwargs.get('return_type','tuple')
        precision = kwargs.get('precision', 2)

        # check which Mode is relevant
        tau_21c = self.r_pl*np.sqrt(1+2*self.p_ppm)
        mode = None

        fe_0 = 0
        fe_fb = self.get_fiber_exertion(stress_vector=stress_vector, precision=precision)

        if stress_vector[1] >= 0:
            mode = 'A'
            fe_0 = np.sqrt((1 - self.p_plp*self.r_pp/self.r_pl)**2*(stress_vector[1]/self.r_pp)**2 + (stress_vector[2]/self.r_pl)**2) + self.p_plp*stress_vector[1]/self.r_pl

        elif stress_vector[1] < 0:
            if stress_vector[2] == 0:
                mode = 'C'
                fe_0 = ((stress_vector[2] / (2 * (1 + self.p_ppm) * self.r_pl)) ** 2 +
                        (stress_vector[1] / self.r_pm) ** 2) * self.r_pm / -stress_vector[1]
            else:
                crit1 = np.abs(stress_vector[1] / stress_vector[2])
                crit2 = np.abs(self.r_ppa / tau_21c)
                if 0 <= crit1 <= crit2:
                    mode = 'B'
                    fe_0 = np.sqrt((stress_vector[2]/self.r_pl)**2 + (self.p_plm/self.r_pl*stress_vector[1])**2) + self.p_plm/self.r_pl*stress_vector[1]
                elif 0 <= 1/crit1 <= 1/crit2:
                    mode = 'C'
                    fe_0 = ((stress_vector[2]/(2*(1 + self.p_ppm)*self.r_pl))**2 + (stress_vector[1]/self.r_pm)**2)*self.r_pm/-stress_vector[1]

        if fe_fb == 0 or fe_0 == 0:
            fe_1 = fe_0

        elif fe_fb/fe_0 < s or fe_0/fe_fb < m:
            fe_1 = fe_0
        else:
            a = (1 - s) / np.sqrt(1 - m ** 2)
            c = fe_0/fe_fb
            eta_w = (c*(a*np.sqrt(c**2*(a**2 - s**2) + 1) +s)/((c*a)**2 + 1))
            fe_1 = fe_0/eta_w

        if precision is not None:
            fe_0 = round(fe_0, precision)
            fe_1 = round(fe_1, precision)

        if ret_type == 'tuple':
            return fe_fb, round(fe_0, 2), round(fe_1, 2), mode
        elif ret_type == 'dict':
            return dict(fe_fb=fe_fb, fe_0=round(fe_0, 2), fe_1=round(fe_1, 2), mode=mode)
        elif ret_type == 'array':
            return np.array([fe_fb, round(fe_0, 2), round(fe_1, 2)]), mode
        else:
            return fe_fb, round(fe_0, 2), round(fe_1, 2), mode


def get_laminate_exertions(laminate, line_loads, *args, **kwargs):
    r"""
    calculates the material exertion of every layer of the laminate from the laminate loads.

    The provided laminate loads (line_loads) are presumed to be line loads in :math:`\frac{N}{mm}` and line moments in
    :math:`N`, defined in the laminate coordinate system. The function returns maximum values for the fiber and
    inter-fiber exertions of all layers as well as a list of dicts, holding details for every layer.
    Every dictionary in the results list has the following entries:

        - fb: fiber-exertion
        - zfb_0: inter_fiber exertion without the influence of fiber parallel stress
        - zfb_1: inter_fiber exertion including the influence of fiber parallel stress
        - mode: the fracture mode as defined by Puck
        - stress: The stress in the layer in layer coordinates
        - strain: The strain in the layer in layer coordinates

    :param laminate: a laminate object
    :param line_loads: an array of line loads
    :type laminate: Laminate
    :type line_loads: numpy.ndarray
    :return: a tuple consisting of

        - maximum fiber exertion in any layer in the laminate
        - maximum inter-fiber exertion in any layer in the laminate
        - a list of dicts holding the results of every layer
    :rtype: tuple
    """
    result = list()
    laminate_strains = laminate.get_strains(line_loads=line_loads)
    layer_strains = laminate.get_layer_strains(laminate_strains=laminate_strains)
    max_fb = 0
    max_zfb = 0

    for layer, (layer_strain_top, layer_strain_bottom) in zip(laminate.stacking, layer_strains):
        layer_dict = dict()
        layer_puck_set = PuckStrengthSet(material=layer.material)
        layer_dict['puck'] = layer_puck_set

        # Bottom of layer
        s = np.dot(layer.material.stiffness_matrix_2d, layer_strain_bottom)
        layer_stress_bottom = np.squeeze(np.asarray(s))
        fe_fb_bottom, fe_0_bottom, fe_1_bottom, mode_bottom = layer_puck_set.puck_exertion_2d(stress_vector=layer_stress_bottom)
        layer_dict['bottom'] = dict(
            fb=fe_fb_bottom, zfb_0=fe_0_bottom, zfb_1=fe_1_bottom, mode=mode_bottom, stress=layer_stress_bottom,
            strain=layer_strain_bottom
        )
        if fe_fb_bottom > max_fb:
            max_fb = fe_fb_bottom

        if fe_1_bottom > max_zfb:
            max_zfb = fe_1_bottom

        # Top of layer
        s = np.dot(layer.material.stiffness_matrix_2d, layer_strain_top)
        layer_stress_top = np.squeeze(np.asarray(s))
        fe_fb_top, fe_0_top, fe_1_top, mode_top = layer_puck_set.puck_exertion_2d(stress_vector=layer_stress_top)
        layer_dict['top'] = dict(
            fb=fe_fb_top, zfb_0=fe_0_top, zfb_1=fe_1_top, mode=mode_top, stress=layer_stress_top,
            strain=layer_strain_top
        )
        if fe_fb_top > max_fb:
            max_fb = fe_fb_top

        if fe_1_top > max_zfb:
            max_zfb = fe_1_top

        result.append(layer_dict)
    return max_fb, max_zfb, result


def find_min_stress_angle(func, *args, **kwargs):
    """
    Find the minimal value of the provided function

    :param func: function to be minimized
    :param precision: (optional) sets the number of decimal points to which the angle is to be calculated, default is 0
    :param offset: (optional) sets the offset used in the provided func, default is 1.0
    :type func: function
    :type precision: int
    :type offset: float
    :return:
    """

    precision = kwargs.get('precision', 0)
    offset = kwargs.get('offset', 1.0)

    thetas = np.linspace(-90, 90, 19)
    values = [func(theta) for theta in thetas]

    p = 0
    delta = 10**-p
    while p <= precision:
        minimum_angel = thetas[np.argmin(values)]
        thetas = np.linspace(minimum_angel - 10 * delta, minimum_angel + 10 * delta, 21)
        values = [func(theta) for theta in thetas]
        p += 1
        delta = 10 ** -p

    return 1. / min(values) - offset, round(thetas[np.argmin(values)], precision)


def get_stress_transformation_matrix(theta, theta_in_deg=False):
    """
    returns a stress transformation matrix that transforms a stress vector from fiber coordinates into a strength plane
    with a normal orientation perpendicular to the fiber orientation. Theta defines the angle to the perpendicular orientation
    in the laminate plane.

    :param theta: rotation angle
    :param theta_in_deg: flag for parameter theta. If true unit of theta is assumed to be degree if not radians.
    :type theta: float
    :type theta_in_deg: bool
    :return: transformation matrix
    :rtype: numpy.ndarray
    """

    if theta_in_deg:
        theta = np.deg2rad(theta)

    trans_matrix = np.zeros((3, 5), dtype=float)
    c = np.cos(theta)
    s = np.sin(theta)

    trans_matrix[0, 0] = c ** 2
    trans_matrix[0, 1] = s ** 2
    trans_matrix[0, 2] = 2 * c * s

    trans_matrix[1, 0] = -c * s
    trans_matrix[1, 1] = c * s
    trans_matrix[1, 2] = c ** 2 - s ** 2

    trans_matrix[2, 3] = s
    trans_matrix[2, 4] = c

    return trans_matrix

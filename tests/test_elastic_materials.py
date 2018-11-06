import pytest
import numpy as np


@pytest.mark.parametrize(('user_input', 'expected'),
                         (
                                 (1, 100000), (11, 100000), ('1', 100000), ('11', 100000),
                                 (None, 100000), ('x', 100000), ('X', 100000),
                                 (2, 8000), (22, 8000), ('2', 8000), ('22', 8000), ('y', 8000), ('Y', 8000),
                                 (3, 7000), (33, 7000), ('3', 7000), ('33', 7000), ('z', 7000), ('Z', 7000),
                                 (12, 5000), (21, 5000), ('12', 5000), ('21', 5000), ('xy', 5000), ('yx', 5000),
                                 ('XY', 5000), ('YX', 5000),
                                 (13, 5000), (31, 5000), ('13', 5000), ('31', 5000), ('xz', 5000), ('zx', 5000),
                                 ('XZ', 5000), ('ZX', 5000),
                                 (23, 4000), (32, 4000), ('23', 4000), ('32', 4000), ('yz', 4000), ('zy', 4000),
                                 ('YZ', 4000), ('ZY', 4000),
                         ))
def test_get_stiffness(ortho_mat, user_input, expected):
    actual = ortho_mat.get_stiffness(user_input)
    assert expected == actual


def test_get_poisson(frp_cfk, fiber_carbon_ht, matrix_ep):
    phi = frp_cfk.fiber_volume_fraction
    assert frp_cfk.get_poisson(12) == round(phi*fiber_carbon_ht.get_poisson(12) + (1-phi) * matrix_ep.get_poisson(12), 2)


@pytest.mark.parametrize(
    ('index', 'direction', 'expected'),
    (
            (11, '+', 1200), ('11', '+', 1200), (11, 'tensile', 1200), ('x', 'tensile', 1200),
            (11, '-', 900), ('11', '-', 900), (11, 'compression', 900), ('x', 'compression', 900),
            (22, '+', 50), ('22', '+', 50), (22, 'tensile', 50), ('y', 'tensile', 50),
            (22, '-', 170), ('22', '-', 170), (22, 'compression', 170), ('y', 'compression', 170),
            (33, '+', 50), ('33', '+', 50), (33, 'tensile', 50), ('z', 'tensile', 50),
            (33, '-', 170), ('33', '-', 170), (33, 'compression', 170), ('z', 'compression', 170),
            (12, None, 70), ('12', None, 70), ('xy', None, 70),
            (21, None, 70), ('21', None, 70), ('yx', None, 70),
            (13, None, 70), ('13', None, 70), ('xz', None, 70),
            (31, None, 70), ('31', None, 70), ('zx', None, 70),
            (23, None, None), ('23', None, None), ('23', None, None),
            (32, None, None), ('32', None, None), ('32', None, None),
    )
)
def test_get_strength(gfk_mat, index, direction, expected):
    actual = gfk_mat.get_strength(index=index, direction=direction)
    assert expected == actual


class TestDensity:
    def test_gfk(self, gfk_mat):
        assert gfk_mat.density == 2.0

    def test_cfk(self, cfk_mat):
        assert cfk_mat.density == 1.5


def test_stiffness_matrix_calculation_cfk(cfk_mat):
    expected = np.array(
        [
            [142620, 4366, 4366, 0, 0, 0],
            [4366, 10561, 3992, 0, 0, 0],
            [4366, 3992, 10561, 0, 0, 0],
            [0, 0, 0, 3285, 0, 0],
            [0, 0, 0, 0, 4600, 0],
            [0, 0, 0, 0, 0, 4600]
        ]
    )
    actual = np.round(cfk_mat.stiffness_matrix, 0)
    assert np.allclose(expected, actual)


def test_stiffness_matrix_calculation_gfk(gfk_mat):
    expected = np.array(
        [
            [49873, 7855, 7855, 0, 0, 0],
            [7855, 18418, 7766, 0, 0, 0],
            [7855, 7766, 18418, 0, 0, 0],
            [0, 0, 0, 5326, 0, 0],
            [0, 0, 0, 0, 5300, 0],
            [0, 0, 0, 0, 0, 5300]
        ]
    )

    actual = np.round(gfk_mat.stiffness_matrix, 0)
    assert np.allclose(expected, actual)


def test_abd(fzb_lam):
    expected = np.array(
        [
            [129903, 37696, 0, 0, 0, 0],
            [37696, 129903, 0, 0, 0, 0],
            [0, 0, 46103, 0, 0, 0],
            [0, 0, 0, 26101, 16591, 10979],
            [0, 0, 0, 16591, 52450, 10979],
            [0, 0, 0, 10979, 10979, 19393]
        ]
    )

    actual = np.round(fzb_lam.abd_matrix, 0)

    assert np.allclose(expected, actual)


def test_abd_async(asym_lam):
    expected = np.array(
        [
            [160188, 26249, 26018, -53409, -200, -4137],
            [26249, 66736, 12146, -200, 17346, 1172],
            [26018, 12146, 32598, -4137, 1172, -1383],
            [-53409, -200, -4137, 59903, 8681, 10054],
            [-200, 17346, 1172, 8681, 20329, 5275],
            [-4137, 1172, -1383, 10054, 5275, 10751]
        ]
    )

    actual = np.round(asym_lam.abd_matrix, 0)
    assert np.allclose(expected, actual)

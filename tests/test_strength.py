from material_mechanics.strength import puck
import numpy as np


def test_puck_laminate_2d(fzb_lam, puck_set_frp_cfk, line_load=np.array([1000, 0, 0, 0, 0, 0])):
    # expected values
    exp_max_fb = 0.63
    exp_max_zfb = 1.2
    exp_exertions = [
        dict(
            bottom={'fb': 0.23, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                    'stress': np.array([459.6209526, 40.3790474, -75.36657208]),
                    'strain': np.array([0.0029833, 0.0029833, -0.01084519])},
            top={'fb': 0.23, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                 'stress': np.array([459.6209526, 40.3790474, -75.36657208]),
                 'strain': np.array([0.0029833, 0.0029833, -0.01084519])}
        ),
        dict(
            bottom={'fb': 0.21, 'zfb_0': 1.2, 'zfb_1': 1.2, 'mode': 'A',
                    'stress': np.array([-3.46030140e+02, 8.39948102e+01, -6.45224969e-15]),
                    'strain': np.array([-2.43929637e-03, 8.40589424e-03, -9.28473670e-19])},
            top={'fb': 0.21, 'zfb_0': 1.2, 'zfb_1': 1.2, 'mode': 'A',
                 'stress': np.array([-3.46030140e+02, 8.39948102e+01, -6.40112526e-15]),
                 'strain': np.array([-2.43929637e-03, 8.40589424e-03, -9.21116903e-19])}
        ),
        dict(
            bottom={'fb': 0.22, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                    'stress': np.array([459.6209526, 40.3790474, 75.36657208]),
                    'strain': np.array([0.0029833, 0.0029833, 0.01084519])},
            top={'fb': 0.22, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                 'stress': np.array([459.6209526, 40.3790474, 75.36657208]),
                 'strain': np.array([0.0029833, 0.0029833, 0.01084519])}
        ),
        dict(
            bottom={'fb': 0.63, 'zfb_0': 0.0, 'zfb_1': 0.0, 'mode': 'C',
                    'stress': np.array([1.26527205e+03, -3.23671540e+00, -2.67524455e-15]),
                    'strain': np.array([8.40589424e-03, -2.43929637e-03, -3.84965593e-19])},
            top={'fb': 0.63, 'zfb_0': 0.0, 'zfb_1': 0.0, 'mode': 'C',
                 'stress': np.array([1.26527205e+03, -3.23671540e+00, -2.72636899e-15]),
                 'strain': np.array([8.40589424e-03, -2.43929637e-03, -3.92322360e-19])}
        ),
        dict(
            bottom={'fb': 0.63, 'zfb_0': 0.0, 'zfb_1': 0.0, 'mode': 'C',
                    'stress': np.array([1.26527205e+03, -3.23671540e+00, -2.62412011e-15]),
                    'strain': np.array([8.40589424e-03, -2.43929637e-03, -3.77608826e-19])},
            top={'fb': 0.63, 'zfb_0': 0.0, 'zfb_1': 0.0, 'mode': 'C',
                 'stress': np.array([1.26527205e+03, -3.23671540e+00, -2.67524455e-15]),
                 'strain': np.array([8.40589424e-03, -2.43929637e-03, -3.84965593e-19])}
        ),
        dict(
            bottom={'fb': 0.22, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                    'stress': np.array([459.6209526, 40.3790474, 75.36657208]),
                    'strain': np.array([0.0029833, 0.0029833, 0.01084519])},
            top={'fb': 0.22, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                 'stress': np.array([459.6209526, 40.3790474, 75.36657208]),
                 'strain': np.array([0.0029833, 0.0029833, 0.01084519])}
        ),
        dict(
            bottom={'fb': 0.21, 'zfb_0': 1.2, 'zfb_1': 1.2, 'mode': 'A',
                    'stress': np.array([-3.46030140e+02, 8.39948102e+01, -6.70787189e-15]),
                    'strain': np.array([-2.43929637e-03, 8.40589424e-03, -9.65257503e-19])},
            top={'fb': 0.21, 'zfb_0': 1.2, 'zfb_1': 1.2, 'mode': 'A',
                 'stress': np.array([-3.46030140e+02, 8.39948102e+01, -6.65674745e-15]),
                 'strain': np.array([-2.43929637e-03, 8.40589424e-03, -9.57900737e-19])}
        ),
        dict(
            bottom={'fb': 0.23, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                    'stress': np.array([459.6209526, 40.3790474, -75.36657208]),
                    'strain': np.array([0.0029833, 0.0029833, -0.01084519])},
            top={'fb': 0.23, 'zfb_0': 0.98, 'zfb_1': 0.98, 'mode': 'A',
                 'stress': np.array([459.6209526, 40.3790474, -75.36657208]),
                 'strain': np.array([0.0029833, 0.0029833, -0.01084519])}
        ),
    ]

    # actual values
    max_fb, max_zfb, laminate_exertion = puck.get_laminate_exertions(laminate=fzb_lam, line_loads=line_load)

    assert max_fb == exp_max_fb
    assert max_zfb == exp_max_zfb
    for actual, expected in zip(laminate_exertion, exp_exertions):
        for position in ('bottom', 'top'):
            for key, val in actual[position].items():
                if key == 'mode':
                    assert val == expected[position][key]
                else:
                    assert np.allclose(val, expected[position][key])


def test_puck_3d(frp_cfk, stress=np.array([1500, 50, 33, 10, 22, 10])):
    # expected values
    fb_expected = 0.75
    zfb_expected = 0.81
    theta_expected = 28.0

    # actual values
    puck_set = puck.PuckStrengthSet(material=frp_cfk)
    fb_actual = puck_set.get_fiber_exertion(stress_vector=stress)
    zfb_actual, theta_actual = puck_set.get_max_inter_fiber_exertion(stress_vector=stress, angle_precision=1)

    # assertions
    assert fb_actual == fb_expected
    assert zfb_actual == zfb_expected
    assert theta_actual == theta_expected


def test_puck_2d(frp_cfk, stress=np.array([1500, 50, 10])):
    # expected values
    fb_expected = 0.75
    zfb0_expected = 0.72
    zfb1_expected = 0.89
    mode_expected = 'A'

    # actual values
    puck_set = puck.PuckStrengthSet(material=frp_cfk)
    fb_actual, zfb0_actual, zfb1_actual, mode_actual = puck_set.puck_exertion_2d(stress_vector=stress)

    # assertions
    assert fb_actual == fb_expected
    assert zfb0_actual == zfb0_expected
    assert zfb1_actual == zfb1_expected
    assert mode_actual == mode_expected

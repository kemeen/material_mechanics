import pytest
import numpy as np
import material_mechanics.tools as funcs


@pytest.mark.parametrize(('symmetry', 's12', 's13', 's21', 's23', 's31', 's32'),
                         ((None, 2720, 2086, 2754, 218, 3142, 2409),
                          ('upper', 2720, 2086, 2720, 218, 2086, 218),
                          ('lower', 2754, 3142, 2754, 2409, 3142, 2409),
                          ('mean', 2737, 2614, 2737, 1313.5, 2614, 1313.5)))
def test_symmetrie_func(ortho_mat, symmetry, s12, s13, s21, s23, s31, s32):
    expected = np.array(
        [
            [101586, s12, s13, 0, 0, 0],
            [s21, 8127, s23, 0, 0, 0],
            [s31, s32, 7111, 0, 0, 0],
            [0, 0, 0, 4000, 0, 0],
            [0, 0, 0, 0, 5000, 0],
            [0, 0, 0, 0, 0, 5000]
        ]
    )
    print(np.round(ortho_mat.stiffness_matrix))
    actual = np.round(funcs.force_symmetry(matrix=ortho_mat.stiffness_matrix, symmetry=symmetry), 1)
    assert np.allclose(expected, actual, atol=0.5)

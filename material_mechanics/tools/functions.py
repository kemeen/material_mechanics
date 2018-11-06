import logging
import numpy as np

# ====================
# Set up module logger
# ====================
logger = logging.getLogger(__name__)


def get_stress_transformation_matrix(theta):
    """
    returns a stress transformation matrix that transforms a stress vector from fiber coordinates into a strength plane
    with a normal orientation perpendicular to the fiber orientation. Theta defines the angle to the perpendicular orientation
    in the laminate plane.

    :param theta: rotation angle
    :return: transformation matrix (numpy.ndarray)
    """
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


def get_T_strain_2d(theta):
    """
    returns a 2d strain transformation matrix that transforms a global strain vector to a strain vector in
    fiber coordinates with a normal orientation perpendicular to the fiber orientation.
    Theta defines the angle to the perpendicular orientation in the laminate plane.

    :param theta: rotation angle
    :return: transformation matrix (numpy.ndarray)
    """
    trans_matrix = np.eye(3, 3, dtype=float)
    c = np.cos(theta)
    c2 = np.cos(2 * theta)
    s = np.sin(theta)
    s2 = np.sin(2 * theta)

    trans_matrix[0, 0] = c ** 2
    trans_matrix[0, 1] = s ** 2
    trans_matrix[0, 2] = 0.5 * s2

    trans_matrix[1, 0] = s ** 2
    trans_matrix[1, 1] = c ** 2
    trans_matrix[1, 2] = -0.5 * s2

    trans_matrix[2, 0] = -s2
    trans_matrix[2, 1] = s2
    trans_matrix[2, 2] = c2

    return trans_matrix


def get_T_strain_3d(theta):
    """
    returns a 3d strain transformation matrix that transforms a global strain vector to a strain vector in
    fiber coordinates with a normal orientation perpendicular to the fiber orientation.
    Theta defines the angle to the perpendicular orientation in the laminate plane.

    :param theta: rotation angle
    :return: transformation matrix (numpy.ndarray)
    """
    trans_matrix = np.eye(6, 6, dtype=float)
    c = np.cos(theta)
    c2 = np.cos(2 * theta)
    s = np.sin(theta)
    s2 = np.sin(2 * theta)

    trans_matrix[0, 0] = c ** 2
    trans_matrix[0, 1] = s ** 2
    trans_matrix[0, 6] = 0.5 * s2

    trans_matrix[1, 0] = s ** 2
    trans_matrix[1, 1] = c ** 2
    trans_matrix[1, 6] = -0.5 * s2

    trans_matrix[6, 0] = -s2
    trans_matrix[6, 1] = s2
    trans_matrix[6, 6] = c2

    return trans_matrix


def get_strain_at_z(global_strains, z):
    """
    returns the local strain interpolated from two given stresses

    :param global_strains:
    :param z: location at which stress should be calculated

    :return: Stress at z location
    """
    eps_0 = np.array(global_strains[:3])
    kappa_0 = np.array(global_strains[3:])

    return eps_0 + z * kappa_0


def force_symmetry(matrix, symmetry):
    r"""
    Enforce symmetry in a given matrix

    :param matrix: matrix with equal number of rows and columns
    :param symmetry: method of symmetry enforcement.

        Options:

        - None: No symmetry is being enforced
        - 'upper': upper-right elements are mirrored to lower-left elements (:math:`n_{ij} = n_{ji}; \; if: i>j`)
        - 'upper': lower-left elements are mirrored to upper-right elements (:math:`n_{ij} = n_{ji}; \; if: i<j`)
        - 'upper': upper-right elements are mirrored to lower-left elements (:math:`n_{ij} = \frac{n_{ji}+n_{ij}}{2}; \; if: i \neq j`)

    :type matrix: numpy.ndarray
    :type symmetry: str or None
    :return:
    """
    symmetric_matrix = matrix.copy()

    if symmetry is None:
        return symmetric_matrix

    for index, x in np.ndenumerate(matrix):

        if symmetry == 'upper':
            if index[0] > index[1]:
                symmetric_matrix[index] = matrix[tuple(reversed(index))]

        if symmetry == 'lower':
            if index[0] < index[1]:
                symmetric_matrix[index] = matrix[tuple(reversed(index))]

        if symmetry == 'mean':
            if index[0] != index[1]:
                symmetric_matrix[index] = np.mean((matrix[index], matrix[tuple(reversed(index))]))

    return symmetric_matrix

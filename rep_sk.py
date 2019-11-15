import math
import vtk
from dipy.data import get_sphere
from vtk.util import numpy_support
import numpy as np
from fury import actor, window, ui
from fury.utils import (numpy_to_vtk_points, numpy_to_vtk_colors,
                        set_polydata_colors, set_polydata_normals,
                        update_polydata_normals, set_input, rotate,
                        set_polydata_vertices, get_actor_from_polydata,
                        set_polydata_triangles)

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def cart2sphere(x, y, z):
    r""" Return angles for Cartesian 3D coordinates `x`, `y`, and `z`
    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.
    $0\le\theta\mathrm{(theta)}\le\pi$ and $-\pi\le\phi\mathrm{(phi)}\le\pi$
    Parameters
    ------------
    x : array_like
       x coordinate in Cartesian space
    y : array_like
       y coordinate in Cartesian space
    z : array_like
       z coordinate
    Returns
    ---------
    r : array
       radius
    theta : array
       inclination (polar) angle
    phi : array
       azimuth angle
    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.divide(z, r, where=r > 0))
    theta = np.where(r > 0, theta, 0.)
    phi = np.arctan2(y, x)
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return r, theta, phi


def vector_norm(vec, axis=-1, keepdims=False):
    """ Return vector Euclidean (L2) norm
    See :term:`unit vector` and :term:`Euclidean norm`
    Parameters
    -------------
    vec : array_like
        Vectors to norm.
    axis : int
        Axis over which to norm. By default norm over last axis. If `axis` is
        None, `vec` is flattened then normed.
    keepdims : bool
        If True, the output will have the same number of dimensions as `vec`,
        with shape 1 on `axis`.
    Returns
    ---------
    norm : array
        Euclidean norms of vectors.
    Examples
    --------
    >>> import numpy as np
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> vector_norm(vec)
    array([ 17.,  85.])
    >>> vector_norm(vec, keepdims=True)
    array([[ 17.],
           [ 85.]])
    >>> vector_norm(vec, axis=0)
    array([  8.,  39.,  77.])
    """
    vec = np.asarray(vec)
    vec_norm = np.sqrt((vec * vec).sum(axis))
    if keepdims:
        if axis is None:
            shape = [1] * vec.ndim
        else:
            shape = list(vec.shape)
            shape[axis] = 1
        vec_norm = vec_norm.reshape(shape)
    return vec_norm


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    Code modified from the work of Christoph Gohlke link provided here
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    Parameters
    ------------
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    Returns
    ---------
    matrix : ndarray (4, 4)
    Code modified from the work of Christoph Gohlke link provided here
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    Examples
    --------
    >>> import numpy
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    _ = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    _ = euler_matrix(ai, aj, ak, axes)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def square():
    vertices = np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0]])
    triangles = np.array([[0, 1, 2],
                          [2, 3, 0]], dtype='i8')
    return vertices, triangles


def box():
    vertices = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 1.0, 1.0],
                         [1.0, 0.0, 0.0],
                         [1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0],
                         [1.0, 1.0, 1.0]])
    triangles = np.array([[0, 6, 4],
                          [0, 2, 6],
                          [0, 3, 2],
                          [0, 1, 3],
                          [2, 7, 6],
                          [2, 3, 7],
                          [4, 6, 7],
                          [4, 7, 5],
                          [0, 4, 5],
                          [0, 5, 1],
                          [1, 5, 7],
                          [1, 7, 3]], dtype='i8')
    return vertices, triangles


def octahedron():
    vertices = np.array([[1.0, 0.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0,  0.0, 1.0],
                         [0.0,  0.0, -1.0]])
    triangles = np.array([[0, 4, 2],
                          [1, 5, 3],
                          [4, 2, 1],
                          [5, 3, 0],
                          [1, 4, 3],
                          [0, 5, 2],
                          [0, 4, 3],
                          [1, 5, 2], ], dtype='i8')
    return vertices, triangles


def icosahedron():
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array([[t,  1,  0],     # 0
                         [-t,  1,  0],    # 1
                         [t, -1,  0],     # 2
                         [-t, -1,  0],    # 3
                         [1,  0,  t],     # 4
                         [1,  0, -t],     # 5
                         [-1,  0,  t],    # 6
                         [-1,  0, -t],    # 7
                         [0,  t,  1],     # 8
                         [0, -t,  1],     # 9
                         [0,  t, -1],     # 10
                         [0, -t, -1], ])   # 11

    vertices /= vector_norm(vertices, keepdims=True)
    triangles = np.array([[8,  4,  0],
                          [2,  5,  0],
                          [2,  5, 11],
                          [9,  2, 11],
                          [2,  4,  0],
                          [9,  2,  4],
                          [10,  8,  1],
                          [10,  8,  0],
                          [10,  5,  0],
                          [6,  3,  1],
                          [9,  6,  3],
                          [6,  8,  1],
                          [6,  8,  4],
                          [9,  6,  4],
                          [7, 10,  1],
                          [7, 10,  5],
                          [7,  3,  1],
                          [7,  3, 11],
                          [9,  3, 11],
                          [7,  5, 11], ], dtype='i8')
    return vertices, triangles


def superquadric(sq_params=np.array([1, 1, 1, 3, 0.001])):
    def _fexp(x, p):
        """a different kind of exponentiation"""
        return (np.sign(x)*(np.abs(x)**p))

    def _superq(sq_params, phi, theta):
        A, B, C, P, Q = sq_params
        x = A * (_fexp(np.sin(phi), P)) * (_fexp(np.cos(theta), Q))
        y = B * (_fexp(np.sin(phi), P)) * (_fexp(np.sin(theta), Q))
        z = C * (_fexp(np.cos(phi), P))
        xyz = np.vstack([x, y, z]).T
        return np.ascontiguousarray(xyz)

    sphere = get_sphere(name='symmetric362')

    vertices = sphere.vertices
    triangles = sphere.faces

    xyz = _superq(sq_params, phi=sphere.theta, theta=sphere.phi)
    vertices = xyz

    return vertices, triangles


def custom_glyph(centers, directions=None, colors=(1, 0, 0),
                 normals=(1, 0, 0), sq_params=None,
                 geom='square', scale=1, **kwargs):
    """Return a custom glyph actor
    """
    if geom.lower() == 'square':
        unit_verts, unit_triangles = square()
        origin_z = 0
    elif geom.lower() == 'box':
        unit_verts, unit_triangles = box()
        origin_z = 0.5
    elif geom.lower() == 'octahedron':
        unit_verts, unit_triangles = octahedron()
        origin_z = 0.5
    elif geom.lower() == 'icosahedron':
        unit_verts, unit_triangles = icosahedron()
        origin_z = 0.5
    elif geom.lower() == 'superquadric':
        unit_verts, unit_triangles = superquadric(sq_params)
        origin_z = 0.5
    else:
        unit_verts, unit_triangles = None
        origin_z = 0

    # update vertices
    big_vertices = np.tile(unit_verts, (centers.shape[0], 1))
    big_centers = np.repeat(centers, unit_verts.shape[0], axis=0)
    # center it
    big_vertices -= np.array([0.5, 0.5, origin_z])
    # apply centers position
    big_vertices += big_centers
    # scale them
    if isinstance(scale, (list, tuple, np.ndarray)):
        scale = np.repeat(scale, unit_verts.shape[0], axis=0)
        scale = scale.reshape((big_vertices.shape[0], 1))
    big_vertices *= scale

    # update triangles
    big_triangles = np.tile(unit_triangles, (centers.shape[0], 1))
    z = np.repeat(np.arange(0, centers.shape[0] *
                            unit_verts.shape[0], step=unit_verts.shape[0]),
                            unit_triangles.shape[0],
                            axis=0).reshape((big_triangles.shape[0], 1))
    big_triangles = np.add(z, big_triangles, casting="unsafe")

    # update colors
    if isinstance(colors, (tuple, list)):
        colors = np.array([colors] * centers.shape[0])
    big_colors = np.repeat(colors*255, unit_verts.shape[0], axis=0)

    # update normals
    if isinstance(normals, (tuple, list)):
        normals = np.array([normals] * centers.shape[0])
    big_normals = np.repeat(normals, unit_verts.shape[0], axis=0)

    # if isinstance(normals, (tuple, list)):
    #     directions = np.array([directions] * centers.shape[0])
    # big_dirs = np.repeat(normals, unit_verts.shape[0], axis=0)
    r, p, t = cart2sphere(0, 0, 1)
    m = euler_matrix(r, p, t, 'rxzy')
    print(big_vertices)
    big_vertices -= big_centers
    big_vertices = np.dot(m[:3, :3], big_vertices.T).T + big_centers

    # Create a Polydata
    pd = vtk.vtkPolyData()
    set_polydata_vertices(pd, big_vertices)
    set_polydata_triangles(pd, big_triangles)
    set_polydata_colors(pd, big_colors)
    set_polydata_normals(pd, big_normals)
    update_polydata_normals(pd)

    current_actor = get_actor_from_polydata(pd)
    if geom.lower() == 'square':
        current_actor.GetProperty().BackfaceCullingOff()
    return current_actor


if __name__ == "__main__":
    showm = window.ShowManager(size=(1920, 1080),
                               order_transparent=True)
    showm.initialize()
#    showm.scene.add(actor.axes())

    centers = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.]])
    sq_params = np.array([1, 2, 1 ,1 ,1])

    showm.scene.add(custom_glyph(centers=centers,
                                 colors=colors, sq_params=sq_params,
                                 directions=directions, geom='superquadric'))
    # showm.scene.background((1, 1, 1))
    showm.start()

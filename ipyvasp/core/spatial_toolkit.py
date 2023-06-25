from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, Voronoi


def tan_inv(vy, vx):
    """
    Returns full angle from x-axis counter clockwise.

    Parameters
    ----------
    vy : Perpendicular componet of vector including sign.
    vx : Base compoent of vector including sign.
    """
    angle = 0  # Place hodler to handle exceptions
    if vx == 0 and vy == 0:
        angle = 0
    elif vx == 0 and np.sign(vy) == -1:
        angle = 3 * np.pi / 2
    elif vx == 0 and np.sign(vy) == 1:
        angle = np.pi / 2
    else:
        theta = abs(np.arctan(vy / vx))
        if np.sign(vx) == 1 and np.sign(vy) == 1:
            angle = theta
        if np.sign(vx) == -1 and np.sign(vy) == 1:
            angle = np.pi - theta
        if np.sign(vx) == -1 and np.sign(vy) == -1:
            angle = np.pi + theta
        if np.sign(vx) == 1 and np.sign(vy) == -1:
            angle = 2 * np.pi - theta
        if np.sign(vx) == -1 and vy == 0:
            angle = np.pi
        if np.sign(vx) == 1 and vy == 0:
            angle = 2 * np.pi
    return angle


def order(points, loop=True):
    """
    Returns indices of counterclockwise ordered vertices of a plane in 3D.

    Parameters
    ----------
    points : numpy array of shape (N,3) or List[List(len=3)].
    loop : Default is True and appends start point at end to make a loop.

    Example
    -------
    >>> pts = np.array([[1,0,3],[0,0,0],[0,1,2]])
    >>> inds = order(pts)
    >>> pts[inds] # Ordered points
    array([[1, 2, 3],
        [0, 0, 0],
        [1, 0, 3]
        [0, 1, 2]])
    """
    points = np.array(points)  # Make array.
    # Fix points if start point is zero.
    if np.sum(points[0]) == 0:
        points = points + 0.5

    center = np.mean(points, axis=0)  # 3D cent point.
    vectors = points - center  # Relative to center

    ex = vectors[0] / np.linalg.norm(vectors[0])  # i
    ey = np.cross(center, ex)
    ey = ey / np.linalg.norm(ey)  # j

    angles = []
    for i, v in enumerate(vectors):
        vx = np.dot(v, ex)
        vy = np.dot(v, ey)
        angle = tan_inv(vy, vx)
        angles.append([i, angle])

    s_angs = np.array(angles)
    ss = s_angs[s_angs[:, 1].argsort()]  # Sort it.

    if loop:  # Add first at end for completing loop.
        ss = np.concatenate((ss, [ss[0]]))

    return ss[:, 0].astype(int)  # Order indices.


def angle_rad(v1, v2):
    "Returns interier angle between two vectors in radians."
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    dot_p = np.round(np.dot(v1, v2) / norm, 12)
    angle = np.arccos(dot_p)
    return angle


def angle_deg(v1, v2):
    "Returns interier angle between two vectors in degrees."
    return np.degrees(angle_rad(v1, v2))


def outer_side(test_point, plane):
    """
    Returns True if test_point is between plane and origin. Could be used to sample BZ mesh in place of ConvexHull.

    Parameters
    ----------
    test_point : array_like 3D point.
    plane : List of at least three coplanar 3D points.
    """
    outside = True
    p_test = np.array(test_point)
    plane = np.unique(plane, axis=0)  # Avoid looped shape.
    c = np.mean(plane, axis=0)  # center
    _dot_ = np.dot(p_test - c, c)
    if _dot_ < -1e-5:
        outside = False
    return outside


def to_plane(normal, points):
    "Project points to a plane defined by `normal`. shape of normal should be (3,) and of points (N,3)."
    if np.ndim(normal) + 1 != np.ndim(points):
        raise ValueError("Shape of points should be (N,3) and of normal (3,).")
    points = np.array(points)
    nu = normal / np.linalg.norm(normal)  # Normal unit vector
    along_normal = points.dot(nu)
    points = points - along_normal[:, None] * nu  # v - (v.n)n
    return points


def rotation(angle_deg, axis_vec):
    """Get a scipy Rotation object at given `angle_deg` around `axis_vec`.

    Usage
    -----
    >>> rot = rotation(60,[0,0,1])
    >>> rot.apply([1,1,1])
    [-0.3660254  1.3660254  1.]
    """
    axis_vec = np.array(axis_vec) / np.linalg.norm(axis_vec)  # Normalization
    angle_rad = np.deg2rad(angle_deg)
    return Rotation.from_rotvec(angle_rad * axis_vec)


def inside_convexhull(hull, points):
    if not isinstance(hull, ConvexHull):
        raise TypeError("hull must be a scipy.spatial.ConvexHull object")

    if np.shape(points)[-1] != hull.points.shape[-1]:
        raise ValueError("points must have same physical dimension as hull.points")

    # A.shape = (facets, d) and b.shape = (facets, 1)
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    eps = np.finfo(np.float32).eps  # do not use custom tolerance here

    # Points inside CovexHull satifies equation Ax + b <= 0.
    return np.all(A @ points.T + b < eps, axis=0)


def to_R3(basis, points):
    """
    Transforms coordinates of points (relative to non-othogonal basis) into orthogonal space.

    Parameters
    ----------
    basis : array_like
        3x3 matrix with basis vectors as rows like [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]].
    points : array_like
        Nx3 points relative to basis, such as KPOINTS and Lattice Points.


    Conversion formula:
    [x,y,z] = n1*b1 + n2*b2 +n3*b3 = [n1, n2, n3] @ [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]]

    .. note::
        Do not use this function if points are Cartesian or provide identity basis.
    """
    return np.array(points) @ basis


def to_basis(basis, coords):
    """Transforms coordinates of points (relative to othogonal basis) into basis space.

    Parameters
    ---------
    basis : array_like
        3x3 matrix with basis vectors as rows like [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]].
    coords : array_like
        Nx3 points relative to cartesian axes.


    Conversion formula:
    [n1, n2, n3] = [x,y,z] @ inv([[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]])
    """
    return np.array(coords) @ np.linalg.inv(basis)


def get_TM(basis1, basis2):
    """Returns a transformation matrix that gives `basis2` when applied on `basis1`.
    basis are 3x3 matrices with basis vectors as rows like [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]].


    >>> from ipyvasp.core.spatial_toolkit import get_TM
    >>> TM = get_TM(basis1, basis2) # Can be ipyvasp.POSCAR.get_TM(basis2) while basis1 is POSCAR.data.basis
    >>> assert np.allclose(basis2, TM @ basis1)
    >>> Q = P @ TM.T # Transform points from P in basis1 to Q in basis2
    >>> # Both P and Q are N x D matrices where N is number of points and D is dimension of space
    ```
    """
    return to_basis(
        basis2, basis1
    )  # basis1 in basis2 is simply the transformation matrix


def get_bz(basis, loop=True, primitive=False):
    """
    Return required data to construct first Brillouin zone.

    Parameters
    ----------
    basis : array_like, shape (3, 3) in reciprocal space.
    loop : If True, joins the last vertex of a BZ plane to starting vertex in order to complete loop.
    primitive : Defualt is False and returns Wigner-Seitz cell, If True returns parallelipiped of basis in reciprocal space.

    Returns
    -------
    BrZoneData(basis, vertices, faces).

    - You can access special points with `.get_special_points` method or by default from `.specials` property.
    - You can access coordinates of faces with `.faces_coords` property.
    - You can access normals vectors of faces with `.normals` property.
    """
    if np.ndim(basis) != 2 and np.shape(basis) != (3, 3):
        raise ValueError("basis must be a 3x3 matrix with basis vectors as rows.")

    basis = np.array(basis)  # Reads
    b1, b2, b3 = basis  # basis are reciprocal basis
    # Get all vectors for BZ
    if primitive:
        # verts, faces, below are in order, if you cange 1, change all
        verts = np.array(
            [[0, 0, 0], b1, b2, b3, b1 + b2, b1 + b3, b2 + b3, b1 + b2 + b3]
        )
        idx_faces = (  # Face are kept anti-clockwise sorted.
            (0, 1, 5, 3, 0),
            (0, 2, 4, 1, 0),
            (0, 3, 6, 2, 0),
            (2, 6, 7, 4, 2),
            (1, 4, 7, 5, 1),
            (3, 5, 7, 6, 3),
        )
        if loop is False:
            idx_faces = tuple(face[:-1] for face in idx_faces)

    else:
        vectors = []
        for i, j, k in product([0, 1, -1], [0, 1, -1], [0, 1, -1]):
            vectors.append(i * b1 + j * b2 + k * b3)

        vectors = np.array(vectors)
        # Generate voronoi diagram
        vor = Voronoi(vectors)
        faces = []
        vrd = vor.ridge_dict
        for r in vrd:
            if r[0] == 0 or r[1] == 0:
                verts_in_face = np.array([vor.vertices[i] for i in vrd[r]])
                faces.append(verts_in_face)

        verts = [v for vs in faces for v in vs]
        verts = np.unique(verts, axis=0)

        # make faces as indices over vertices because that what most programs accept
        idx_faces = []
        for face in faces:
            vert_inds = [
                i for i, v in enumerate(verts) if tuple(v) in [tuple(f) for f in face]
            ]  # having tuple comparsion is important here.
            idx_faces.append(
                vert_inds
            )  # other zero is to pick single index out of same three

        # order faces
        idx_faces = [
            tuple(face[i] for i in order(verts[face], loop=loop)) for face in idx_faces
        ]

    out_dict = {
        "basis": basis,
        "vertices": verts,
        "faces": idx_faces,
        "primitive": primitive,
    }
    from .serializer import BrZoneData  # to avoid circular import

    return BrZoneData(out_dict)


def kpoints2bz(bz_data, kpoints, shift=0):
    """
    Brings KPOINTS inside BZ. Applies `to_R3` only if bz_data is for primitive BZ.

    Parameters
    ----------
    bz_data : Output of get_bz().
    kpoints : List or array of KPOINTS to transorm into BZ or R3.
    shift : This value is added to kpoints before any other operation, single number of list of 3 numbers for each direction.
    """
    kpoints = np.array(kpoints) + shift
    if bz_data.primitive:
        return to_R3(bz_data.basis, kpoints)

    cent_planes = [
        np.mean(np.unique(face, axis=0), axis=0) for face in bz_data.faces_coords
    ]

    out_coords = np.empty(np.shape(kpoints))  # To store back

    def inside(coord, cent_planes):
        _dots_ = np.max(
            [np.dot(coord - c, c) for c in cent_planes]
        )  # max in all planes
        # print(_dots_)
        if np.max(_dots_) > 1e-8:  # Outside
            return []  # empty for comparison
        else:  # Inside
            return list(coord)  # Must be in list form

    for i, p in enumerate(kpoints):
        for q in product([0, 1, -1], [0, 1, -1], [0, 1, -1]):
            # First translate, then make coords, then feed it back
            # print(q)
            pos = to_R3(bz_data.basis, p + np.array(q))
            r = inside(pos, cent_planes)
            if r:
                # print(p,'-->',r)
                out_coords[i] = r
                StopIteration

    return out_coords  # These may have duplicates, apply np.unique(out_coords,axis=0). do this in surface plots

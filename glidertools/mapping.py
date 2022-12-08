from inspect import currentframe as getframe

import numpy as np

from matplotlib import pyplot as plt
from numexpr import evaluate

from .helpers import GliderToolsWarning, transfer_nc_attrs


class QuadTree:
    """
    Recursively splits data into quadrants.

    Object oriented quadtree can access children recursively

    Properties
    ----------
    children : list
        always 4 QuadTree children
    siblings : list
        always 3 QuadTree siblings
    neighbours : list
        a list of adjacent QuadTrees
    parent : QuadTree
        the current quadtree's parent
    root : QuadTree
        the level 0 quadtree
    is_leaf : bool
        if final QuadTree with no children then True
    n_points : int
        number of points in the quadtree
    location : list
        the quadtree nested location index; e.g. [0, 3, 1]
    depth : int
        the depth of the current quadtree (relative to root)
    max_depth : int
        the deepest level for the current quadtree (relative to parent)
    data : array
        data inside the quadrant
    index : array
        the original indicies [0-N] for data inside the quadrant
        where N is the total number of coordinates given

    """

    def __init__(
        self,
        data,
        mins=None,
        maxs=None,
        max_points_per_quad=50,
        location=[],
        index=None,
        recursion_limit=15,
        parent=None,
    ):

        self.data = np.asarray(data)

        # data should be two-dimensional
        assert self.data.shape[1] == 2

        # duplicate values > max_points will result in recursion error
        # raise an error if there are duplicates in the coordinates
        # self._check_duplicate_values(data[:, 0], max_points_per_quad)
        # self._check_duplicate_values(data[:, 1], max_points_per_quad)

        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)
        if index is None:
            self.index = np.arange(data.shape[0])
        else:
            self.index = index

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.xlim, self.ylim = np.array([mins, maxs]).T
        self.sizes = self.maxs - self.mins
        self.n_points = data.shape[0]
        self.location = location
        self.depth = len(location)
        self.parent = parent
        self.is_leaf = False  # changes later if not

        self.children = []

        mids = 0.5 * (self.mins + self.maxs)
        xmin, ymin = self.mins
        xmax, ymax = self.maxs
        xmid, ymid = mids

        # split the data into four quadrants
        index_q0 = (data[:, 0] <= mids[0]) & (data[:, 1] >= mids[1])
        index_q1 = (data[:, 0] >= mids[0]) & (data[:, 1] >= mids[1])
        index_q2 = (data[:, 0] <= mids[0]) & (data[:, 1] <= mids[1])
        index_q3 = (data[:, 0] >= mids[0]) & (data[:, 1] <= mids[1])

        data_q0 = data[index_q0]
        data_q1 = data[index_q1]
        data_q2 = data[index_q2]
        data_q3 = data[index_q3]

        if self.n_points > max_points_per_quad:
            props = dict(max_points_per_quad=max_points_per_quad, parent=self)
            self.children.append(
                QuadTree(
                    data_q0,
                    [xmin, ymid],
                    [xmid, ymax],
                    index=self.index[index_q0],
                    location=location + [0],
                    **props
                )
            )
            self.children.append(
                QuadTree(
                    data_q1,
                    [xmid, ymid],
                    [xmax, ymax],
                    index=self.index[index_q1],
                    location=location + [1],
                    **props
                )
            )
            self.children.append(
                QuadTree(
                    data_q2,
                    [xmin, ymin],
                    [xmid, ymid],
                    index=self.index[index_q2],
                    location=location + [2],
                    **props
                )
            )
            self.children.append(
                QuadTree(
                    data_q3,
                    [xmid, ymin],
                    [xmax, ymid],
                    index=self.index[index_q3],
                    location=location + [3],
                    **props
                )
            )
        else:
            self.is_leaf = True

    def __getitem__(self, args, fail=True):
        args = np.array(args, ndmin=1)

        if any(args > 3):
            raise UserWarning("A quadtree only has 4 indicies (start locationing at 0)")

        quadtree = self
        passed = []
        for depth in args:
            if (len(quadtree.children) > 0) | fail:
                quadtree = quadtree.children[depth]
                passed += [depth]
            else:

                return None

        return quadtree

    def loc(self, *args, fail=True):
        return self.__getitem__(args, fail=fail)

    def __repr__(self):
        return "<{} : {}>".format(str(self.__class__)[1:-1], str(self.location))

    def __str__(self):

        location = str(self.location)[1:-1]
        location = location if location != "" else "[] - base QuadTree has no location"

        # boundaries and spacing to make it pretty
        left, top = self.mins
        right, bot = self.maxs
        wspace = " " * int(np.log1p(left) + 2)

        # text output (what youll see when you print the object)
        about_tree = "\n".join(
            [
                "",
                "QuadTree object",
                "===============",
                "  location:         {}".format(location),
                "  depth:            {}".format(len(self.location)),
                "  n_points:         {}".format(self.n_points),
                "  boundaries:",
                "      {}{:.2f}".format(wspace, top),
                "    {:.2f}      {:.2f}".format(left, right),
                "      {}{:.2f}".format(wspace, bot),
                "  children_points:  {}".format(
                    str([c.n_points for c in self.children])
                ),
            ]
        )
        return about_tree

    def _traverse_tree(self):
        if not self.children:
            yield self
        for child in self.children:
            yield from child._traverse_tree()

    def query_xy(self, x, y):

        mids = 0.5 * (self.mins + self.maxs)

        idx = np.where(
            [
                (x < mids[0]) & (y > mids[1]),
                (x >= mids[0]) & (y > mids[1]),
                (x < mids[0]) & (y <= mids[1]),
                (x >= mids[0]) & (y <= mids[1]),
            ]
        )[0].tolist()

        self = self.loc(*idx, fail=False)
        while not self.is_leaf:
            self = self.query_xy(x, y)

        return self

    def get_leaves(self):
        return list(set(list(self._traverse_tree())))

    def get_leaves_attr(self, attr):
        return [getattr(q, attr) for q in self.leaves]

    def draw_tree(self, ax=None, depth=None):
        """Recursively plot a visualization of the quad tree region"""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplots(figsize=[11, 7], dpi=150)[1]

        if depth is None or depth == 0:
            rect = plt.Rectangle(
                self.mins, *self.sizes, zorder=2, alpha=1, lw=1, ec="#cccccc", fc="none"
            )
            ax.add_patch(rect)
        if depth is None or depth > 0:
            for child in self.children:
                child.draw_tree(ax, depth - 1)

        return ax

    @property
    def root(self):
        parent = self
        for _ in self.location:
            parent = parent.parent
        return parent

    @property
    def siblings(self):
        if self.location == []:
            return None

        siblings = self.parent.children.copy()
        siblings.remove(self)

        return siblings

    @property
    def neighbours(self):
        def get_border_children(quad, location):
            """Returns all T/L/R/B boundaries as defined by bound_location"""
            bounds = [[2, 3], [0, 2], [0, 1], [1, 3]]
            bound_location = bounds[location]
            if not quad.is_leaf:
                for i in bound_location:
                    yield from get_border_children(quad.children[i], location)
            else:
                yield quad

        neighbours = []

        root = self.root

        ########################
        # IMMEDIATELY ADJACENT #
        coords = [
            (
                self.xlim[0] + self.sizes[0] / 2,
                self.ylim[1] + self.sizes[1] / 2,
            ),
            (
                self.xlim[1] + self.sizes[0] / 2,
                self.ylim[0] + self.sizes[1] / 2,
            ),
            (
                self.xlim[0] + self.sizes[0] / 2,
                self.ylim[0] - self.sizes[1] / 2,
            ),
            (
                self.xlim[0] - self.sizes[0] / 2,
                self.ylim[0] + self.sizes[1] / 2,
            ),
        ]
        # loop through top, right, bottom, left
        for i in range(4):
            x, y = coords[i]
            query_quad = root.query_xy(x, y)
            if query_quad is not None:
                same_size_idx = query_quad.location[: self.depth]
                same_size_quad = root[same_size_idx]
                neighbours += list(get_border_children(same_size_quad, i))

        #############
        # DIAGONALS #
        xs, ys = (root.sizes / 2**root.max_depth) / 2
        neighbours += [
            root.query_xy(self.xlim[0] - xs, self.ylim[0] - ys),  # TL
            root.query_xy(self.xlim[1] + xs, self.ylim[0] - ys),  # TR
            root.query_xy(self.xlim[0] - xs, self.ylim[1] + ys),  # BL
            root.query_xy(self.xlim[1] + xs, self.ylim[1] + ys),  # BR
        ]

        unique_neighbours = list(set(neighbours))
        try:
            unique_neighbours.remove(self)
        except ValueError:
            pass

        return unique_neighbours

    @property
    def max_depth(self):
        leaves = self.get_leaves()
        depths = np.array([leaf.depth for leaf in leaves])

        return depths.max()

    @property
    def leaves(self):
        return self.get_leaves()

    def _plot_quad_test(self):
        from matplotlib import pyplot as plt

        ax = plt.subplots(figsize=[11, 7], dpi=150)[1]

        for depth in range(self.root.max_depth + 1):
            self.root.draw_tree(ax=ax, depth=depth)

        x, y = self.root.data.T
        ax.plot(x, y, c="blue")

        for qa in self.neighbours:
            qa.draw_rectangle(ax, alpha=0.3, lw=0, fc="orange")
            ax.plot(qa.data[:, 0], qa.data[:, 1], "o", ms=1, c="orange")

        ax.set_ylim(ax.get_ylim()[::-1])

        return ax

    @staticmethod
    def _check_duplicate_values(y, max_duplicates=50):
        """
        Checks if a array-like has duplicate values.

        Parameters
        ----------
        y : array-like
            can be any flat array
        max_duplicates : int

        Raises
        ------
        RecursionError with a plot to show where duplicates are
        """
        if y.size < 1:
            return
        x = np.arange(y.size)

        uniq, count = np.unique(y, return_counts=True)
        duplicate = count.max()

        if duplicate >= max_duplicates:
            _, ax = plt.subplots(figsize=[7, 4])
            m = uniq[count == duplicate] == y

            ax.plot(x, y, "-b", lw=2)
            ax.plot(x[m], y[m], "-b", lw=6)

            ax.set_xlabel("Point index")
            ax.set_ylabel("Variable")
            plt.show()

            raise RecursionError(
                "Your data has duplicate coordinates "
                "(as shown by thick line in plot). You "
                "need to remove these points"
            )


def interp_leaf(
    leaf,
    z=None,
    xi=None,
    yi=None,
    lenscale_x=1,
    lenscale_y=15,
    nugget=0.001,
    partial_sill=0.01,
    range=1,
    min_points_per_quad=8,
    return_error=False,
    verbose=True,
):
    """
    Leaf by leaf Kriging interpolation of data. Must use
    GliderTools.mapping.QuadTree leaves as input. z, xi, yi are compulsory
    inputs (but set as keywords for parallel calling)
    """

    def get_leaf_interp_points(leaf, xi, yi):
        """
        Subfunction to get the interpolation points for a leaf
        Helper for main function (useful when leaf has neighbours)

        """
        xj = (xi >= leaf.xlim[0]) & (xi <= leaf.xlim[1])
        yj = (yi >= leaf.ylim[0]) & (yi <= leaf.ylim[1])
        j = xj & yj
        return j

    def calc_neighbour_weights(x, y, limx, limy):
        w = np.ones(x.size)

        x0, x1 = limx
        y0, y1 = limy
        dx = np.diff(limx)
        dy = np.diff(limy)

        iL = x <= x0
        iR = x >= x1
        iB = y <= y0
        iT = y >= y1

        w[iL] = w[iL] / (1 + 16 * ((x0 - x[iL]) / dx) ** 2)
        w[iR] = w[iR] / (1 + 4 * ((x1 - x[iR]) / dx) ** 2)
        w[iB] = w[iB] / (1 + 16 * ((y0 - y[iB]) / dy) ** 2)
        w[iT] = w[iT] / (1 + 4 * ((y1 - y[iT]) / dy) ** 2)

        return w

    ###################################################
    # NEIGHBOURS #
    interp_leaves = [leaf]  # creating a list to append
    for nb in leaf.neighbours:
        interp_leaves += [nb]  # adding neighbour to the list
        # and neighbours' neighbours if low on points
        if nb.n_points < min_points_per_quad:
            interp_leaves += nb.neighbours
    interp_leaves = set(interp_leaves)  # removing duplicates with set

    # BASIS POINTS ###################################
    # xb is the flat basis points
    xb, yb = np.concatenate([el.data for el in interp_leaves]).T
    zb = z[np.concatenate([el.index for el in interp_leaves])]

    # NEIGHBOURHOOD WEIGHTS ###########################
    # note that neighbourhood weights are calculated only with
    # the basis points and not with the interpolation
    ii = [get_leaf_interp_points(el, xi, yi) for el in interp_leaves]
    ii = np.array(ii).any(axis=0)
    xn = xi[ii]
    yn = yi[ii]
    nw = calc_neighbour_weights(xn, yn, leaf.xlim, leaf.ylim)

    zi, ei = _kriging_wrapper(
        xb, yb, zb, xn, yn, lenscale_x, lenscale_y, partial_sill, nugget, range
    )

    # VAR NAMES : nw = neighbour weights, wi = interpolation weights
    zi = nw * zi
    ei = nw * ei

    return nw, zi, ei, ii


def _kriging_wrapper(x, y, z, xi, yi, x_length, y_length, partial_sill, nugget, range):
    """
    I borrowed heavily from PyKrige
    """
    import numpy as np

    def _gaussian(partial_sill, range, nugget, d):
        expr = "partial_sill * (1 - exp(-(d*d) / (range * 4 / 7)**2)) + nugget"
        return evaluate(expr)

    def _get_interp_distances(x, y, xp, yp):
        from scipy.spatial.distance import cdist

        xy = np.c_[x, y]
        xyp = np.c_[xp, yp]

        dist = cdist(xyp, xy)

        return dist

    def _get_kriging_matrix(x, y, variogram_func):
        """Assembles the kriging matrix."""
        from scipy.spatial.distance import cdist

        n = x.size
        xy = np.c_[x, y]
        d = cdist(xy, xy, "euclidean")

        a = np.zeros((n + 1, n + 1))
        a[:n, :n] = -variogram_func(d)

        np.fill_diagonal(a, 0.0)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        return a

    def _exec_vector(a, bd, Z, variogram_func):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""
        import scipy

        eps = 1e-10
        npt = bd.shape[0]
        n = Z.size
        zero_index = None
        zero_value = False

        a_inv = scipy.linalg.inv(a)

        if np.any(np.absolute(bd) <= eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= eps)

        b = np.zeros((npt, n + 1, 1))

        b[:, :-1, 0] = -variogram_func(bd)

        if zero_value:
            b[zero_index[0], zero_index[1], 0] = 0.0
        b[:, n, 0] = 1.0

        x = np.dot(a_inv, b.reshape((npt, n + 1)).T).reshape((1, n + 1, npt)).T
        zvalues = np.sum(x[:, :n, 0] * Z, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return zvalues, sigmasq

    no_nans = ~np.any([np.isnan(a) for a in [x, y, z]])
    assert no_nans, "your input data has nans - remove them first"

    x /= x_length
    y /= y_length
    xi /= x_length
    yi /= y_length

    def func(d):
        return _gaussian(partial_sill, range, nugget, d)

    a = _get_kriging_matrix(x, y, func)
    b = _get_interp_distances(x, y, xi, yi)
    zi, ei = _exec_vector(a, b, z, func)

    return zi, ei


def interp_obj(  # noqa: C901
    x,
    y,
    z,
    xi,
    yi,
    partial_sill=0.1,
    nugget=0.01,
    lenscale_x=20,
    lenscale_y=20,
    detrend=True,
    max_points_per_quad=55,
    min_points_per_quad=8,
    return_error=False,
    n_cpus=None,
    verbose=True,
    parallel_chunk_size=512,
):
    """
    Performs objective interpolation (or Kriging) of a 2D field.

    The objective interpolation breaks the problem into smaller fields by
    iteratively breaking the problem into quadrants. Each quadrant is then
    interpolated (also using intformation from its neighbours).
    The interpolation is inverse distance weighted using a gaussian kernel (or
    radial basis function). The kernel has a width of 12 hours if the
    x-dimension is time, otherwise scaled by the x-variable unit. The kernel
    is in meters assuming that depth is the y-coord. This can be changed with
    keyword arguements. An error estimate can also be calculated if requested.

    The following link provides good background on the Kriging procedure:
    http://desktop.arcgis.com/en/arcmap/10.3/tools/3d-analyst-toolbox/how-kriging-works.htm


    Parameters
    ----------
    x : np.array | pd.series
        horizontal coordinates of the input data (same length as y, z)
        can be types float or datetime64
    y : np.array | pd.series
        vertical coordinates of the input data (same length as x, z)
    z : np.array | pd.series
        values to be interoplated (same length as x, y)
    xi : np.array
        horizontal coordinates of the interpolation grid (must be 1D)
        can be types float or datetime64
    yi : np.array | pd.series
        vertical coordinates of the interpolation grid (must be 1D)
    nugget : float [0.01]
        the error estimate due to sampling inaccuracy also known as the nugget
        in Kriging literature. This should be taken from the semivariogram
    partial_sill : float [0.1]
        represents the spatial covariance of the variable being interpolated.
        Should be estimated from the semivariogram. See Kriging literature for
        more information
    lenscale_x : float [20]
        horizontal length scale horizontal coordinate variable
        If dtype(x) is np.datetime64 (any format) then lenscale units is in
        hours. Otherwise if type(x).
    lenscale_y : float [20]
        horizontal length scale horizontal coordinate variable.
    max_points_per_quad : int [55]
        the data is divided into quadrants using a quadtree approach -
        iteratively dividing data into smaller quadrants using x and y
        coordinates. The algorithm stops splitting the data into quadrants
        when there are no quadrants exceeding the limit set with
        max_points_per_quad is. This is done to reduce the computational
        cost of the function.
    min_points_per_quad : int [8]
        sets the minimum number of points allowed in a neighbouring quadrant
        when creating the interpolation function for a particular quadrant. If
        the number of points is less than specified, the algorithm looks for
        neighbours of the neighbours to include more points in the
        interpolation.
    n_cpus : int [n - 1]
        use parallel computing. The quadrant calculations are spread across
        CPUs. Must be positive and > 0
    parallel_chunk_size : int [512]
        the number of leaves that will be processed in parallel in one go. This
        is a memory saving feature. If your dataset is very large, parallel
        processing will use up a lot of memmory. Increasing the chunk size
        increases the memory requirements.
    verbose : bool [True]
        will print out information about the interpolation

    Returns
    -------
    xr.Dataset
        Contains the following arrays:
        - z: interpolated values
        - variance: error estimate of the interpolation
        - weights: the quadtree weighting used to calculate the estimates
        - nugget: the nugget used in the interpolation
        - partial_sill: value used for the interpolation

    Note
    ----
    The data may have semi-discrete artifacts. This is also present in the
    MATLAB output.

    Example
    -------
    >>> xi = np.arange(time.values.min(), time.values.max(), 30,
                       dtype='datetime64[m]')
    >>> yi = np.linspace(depth.min(), depth.max(), 1.)
    >>> interpolated = gt.mapping.interp_obj(
            time, depth, var, xi, yi,
            nugget=.0035, partial_sill=0.02,
            lenscale_x=80, lenscale_y=80,
            detrend=True)

    """

    def get_detrend_model(x, y, z):
        model = linear_model.LinearRegression()
        model.fit(np.c_[x, y], z)

        return model

    import multiprocessing as mp

    from functools import partial
    from time import perf_counter as timer

    import xarray as xr

    from sklearn import linear_model

    if (n_cpus is None) | (n_cpus == 0):
        n_cpus = mp.cpu_count() - 1

    if verbose:
        print("Starting Interpolation with quadtree optimal interpolation")
        print("----------------------------------------------------------")
        print("\nPreparing for interpolations:")

    zvar = z.copy()
    yvar = y.copy()
    xvar = x.copy()

    is_time_x = np.issubdtype(x.dtype, np.datetime64)
    is_time_xi = np.issubdtype(xi.dtype, np.datetime64)
    ymessage = "y-coordinates are not the same type (x={}, xi={})".format(
        y.dtype, yi.dtype
    )
    xmessage = "x-coordinates are not the same type (x={}, xi={})".format(
        x.dtype, xi.dtype
    )
    assert y.dtype == yi.dtype, ymessage
    assert (is_time_x + is_time_xi) != 1, xmessage

    if is_time_x:  # convert data to hours
        if verbose:
            print("\tTime conversion")
        x = np.array(x).astype("datetime64[s]").astype(float) / 3600
        xi = np.array(xi).astype("datetime64[s]").astype(float) / 3600
        units_x = "hrs"
    else:
        units_x = ""

    if verbose:
        print("\tFinding and removing nans")
    nans = np.isnan(z) | np.isnan(x) | np.isnan(y)
    x, y, z = [np.array(a)[~nans] for a in [x, y, z]]

    # detrend data using linear regression
    if detrend:
        if verbose:
            print("\tRemoving data trend with linear regression")
        model = get_detrend_model(x, y, z)
        z_hat = model.predict(np.c_[x, y])
        z -= z_hat
    else:
        if verbose:
            print("\tRemoving data mean")
        z_avg = np.nanmean(z)
        z -= z_avg

    if verbose:
        print("\tBuilding QuadTree")
    quad_tree = QuadTree(np.c_[x, y], max_points_per_quad=max_points_per_quad)
    xx, yy = np.array(np.meshgrid(xi, yi)).reshape(2, -1)
    leaves = quad_tree.leaves
    n = len(leaves)

    interp_info = "\n".join(
        [
            "\nInterpolation information:",
            "\tbasis points:        {}".format(x.size),
            "\tinterp grid:         {}, {}".format(xi.size, yi.size),
            "\tmax_points_per_quad: {}".format(max_points_per_quad),
            "\tmin_points_per_quad: {}".format(min_points_per_quad),
            "\tnumber of quads:     {}".format(n),
            "\tdetrend_method:      {}".format(
                "linear_regression" if detrend else "mean"
            ),
            "\tpartial_sill:        {}".format(partial_sill),
            "\tnugget:              {}".format(nugget),
            "\tlengthscales:        X = {} {}".format(lenscale_x, units_x),
            "\t                     Y = {} m".format(lenscale_y),
        ]
    )

    if verbose:
        print(interp_info)

    pool = mp.Pool(n_cpus)
    props = dict(
        z=z,
        xi=xx,
        yi=yy,
        nugget=nugget,
        partial_sill=partial_sill,
        lenscale_x=lenscale_x,
        lenscale_y=lenscale_y,
        min_points_per_quad=min_points_per_quad,
        return_error=return_error,
        verbose=verbose,
    )

    func = partial(interp_leaf, **props)

    # predifining matricies for interpolation
    errors = np.ndarray(xx.size) * 0
    weights = np.ndarray(xx.size) * 0
    variable = np.ndarray(xx.size) * 0
    # creating a timer to inform the user
    t0 = timer()
    # getting the index used to split the data up into chunks
    chunk_idx = np.arange(0, n, parallel_chunk_size, dtype=int)
    n_chunks = chunk_idx.size
    if verbose:
        print(
            "\nProcessing interpolation in {} parts over {} CPUs:".format(
                n_chunks, n_cpus
            )
        )
    for c, i0 in enumerate(chunk_idx):
        i1 = i0 + parallel_chunk_size
        chunk_leaves = leaves[i0:i1]
        # do the parallel processing
        chunk_output = pool.map(func, chunk_leaves)
        # add the parallel chunk output to the output arrays
        for w, zi, er, ii in chunk_output:
            weights[ii] += w
            variable[ii] += zi
            errors[ii] += er
        # create info for the user
        t1 = timer()
        if verbose:
            print("\tchunk {}/{} completed in {:.0f}s".format(c + 1, n_chunks, t1 - t0))
        t0 = timer()

    # completing the interpolation
    if verbose:
        print("\nFinishing off interoplation")
    if detrend:
        if verbose:
            print("\tAdding back the trend")
        zi = (variable / weights) + model.predict(np.c_[xx, yy])
    else:
        if verbose:
            print("\tAdding back the average")
        zi = (variable / weights) + z_avg
    errors = errors / weights
    if verbose & is_time_x:
        print("\tTime conversion")
    xi = (xi * 3600).astype("datetime64[s]") if is_time_x else xi

    if verbose:
        print("\tCreating xarray dataset for output")
    xds = xr.Dataset(
        attrs={
            "description": (
                "interpolation output from the GliderTools.interp_obj"
                "function. Print out mapping_info for more details"
            ),
            "mapping_info": interp_info,
        }
    )

    props = dict(dims=["y", "x"], coords={"y": yi, "x": xi})
    xds["z"] = xr.DataArray(zi.reshape(yi.size, xi.size), **props)
    xds["weights"] = xr.DataArray(weights.reshape(yi.size, xi.size), **props)
    xds["variance"] = xr.DataArray(errors.reshape(yi.size, xi.size), **props)
    xds.attrs["nugget"] = nugget
    xds.attrs["partial_sill"] = partial_sill

    dummy = transfer_nc_attrs(getframe(), zvar, zvar, "_interp")
    if isinstance(zvar, xr.DataArray):
        xds["z"].attrs = dummy.attrs
        # xds = xds.rename({'z': dummy.name})

    if isinstance(yvar, xr.DataArray):
        xds["y"].attrs = yvar.attrs
        xds = xds.rename({"y": yvar.name})

    if isinstance(xvar, xr.DataArray):
        xds["x"].attrs = xvar.attrs
        xds = xds.rename({"x": xvar.name})

    return xds


def grid_data(
    x,
    y,
    var,
    bins=None,
    how="mean",
    interp_lim=6,
    verbose=True,
    return_xarray=True,
):
    """
    Grids the input variable to bins for depth/dens (y) and time/dive (x).
    The bins can be specified to be non-uniform to adapt to variable sampling
    intervals of the profile. It is useful to use the ``gt.plot.bin_size``
    function to identify the sampling intervals. The bins are averaged (mean)
    by default but can also be the ``median, std, count``,

    Parameters
    ----------
    x : np.array, dtype=float, shape=[n, ]
        The horizontal values by which to bin need to be in a psudeo discrete
        format already. Dive number or ``time_average_per_dive`` are the
        standard inputs for this variable. Has ``p`` unique values.
    y : np.array, dtype=float, shape=[n, ]
        The vertical values that will be binned; typically depth, but can also
        be density or any other variable.
    bins : np.array, dtype=float; shape=[q, ], default=[0 : 1 : max_depth ]
        Define the bin edges for y with this function. If not defined, defaults
        to one meter bins.
    how : str, defualt='mean'
        the string form of a function that can be applied to pandas.Groupby
        objects. These include ``mean, median, std, count``.
    interp_lim : int, default=6
        sets the maximum extent to which NaNs will be filled.

    Returns
    -------
    glider_section : xarray.DataArray, shape=[p, q]
        A 2D section in the format specified by ``ax_xarray`` input.

    Raises
    ------
    Userwarning
        Triggers when ``x`` does not have discrete values.
    """
    from numpy import array, c_, diff, unique
    from pandas import Series, cut
    from xarray import DataArray

    xvar, yvar = x.copy(), y.copy()
    z = Series(var)
    y = array(y)
    x = array(x)

    u = unique(x).size
    s = x.size
    if (u / s) > 0.2:
        raise UserWarning(
            "The x input array must be psuedo discrete (dives or dive_time). "
            "{:.0f}% of x is unique (max 20% unique)".format(u / s * 100)
        )

    chunk_depth = 50
    # -DB this might not work if the user uses anything other than depth, example
    # density. Chunk_depth would in that case apply to density, which will
    # probably have a range that is much smaller than 50.
    optimal_bins, avg_sample_freq = get_optimal_bins(y, chunk_depth)
    if bins is None:
        bins = optimal_bins

    # warning if bin average is smaller than average bin size
    # -DB this is not being raised as a warning. Instead just seems like useful
    # information conveyed to user. Further none of this works out if y is not
    # depth, since avg_sample freq will not make sense otherwise.
    if verbose:
        avg_bin_size = diff(bins).mean()
        print(
            (
                "Mean bin size = {:.2f}\n"
                "Mean depth binned ({} m) vertical sampling frequency = {:.2f}"
            ).format(avg_bin_size, chunk_depth, avg_sample_freq)
        )

    labels = c_[bins[:-1], bins[1:]].mean(axis=1)  # -DB creates the mean bin values
    bins = cut(y, bins, labels=labels)
    # -DB creates a new variable where instead of variable the bin category
    # is mentioned (sort of like a discretization)

    grp = Series(z).groupby([x, bins])  # -DB put z into the many bins (like 2D hist)
    grp_agg = getattr(
        grp, how
    )()  # -DB basically does grp.how() or in this case grp.mean()
    gridded = grp_agg.unstack(level=0)
    gridded = gridded.reindex(labels.astype(float))

    if interp_lim > 0:
        gridded = gridded.interpolate(limit=interp_lim).bfill(limit=interp_lim)

    if not return_xarray:
        return gridded

    if return_xarray:
        dummy = transfer_nc_attrs(getframe(), var, var, "_vert_binned")

        xda = DataArray(gridded)
        if isinstance(var, DataArray):
            xda.attrs = dummy.attrs
            xda.name = dummy.name

        if isinstance(yvar, DataArray):
            y = xda.dims[0]
            xda[y].attrs = yvar.attrs
            xda = xda.rename({y: yvar.name})

        if isinstance(xvar, DataArray):
            x = xda.dims[1]
            xda[x].attrs = xvar.attrs
            xda = xda.rename({x: xvar.name})

        return xda


def get_optimal_bins(depth, chunk_depth=50, round_up=True):
    """
    Uses depth data to estimate the optimal bin depths for gridding.

    Data is grouped in 50 m chunks (default for chunk_depth) where the
    average sequential depth difference is used to estimate the binning
    resolution for that chunk. The chunk binning resolution is rounded to
    the upper/lower 0.5 metres (specified by user).

    Parameters
    ----------
    depth : array
        A sequential array of depth (concatenated dives)
    chunk_depth : float=50
        chunk depth over which the bin sizes will be calculated
    round_up : True
        if True, rounds up to the nearest 0.5 m, else rounds down.

    Returns
    -------
    bins : array
    bin_avg_sampling_freq : float
        un-rounded depth weighted depth sampling frequency (for verbose use)

    """

    from numpy import abs, arange, array, ceil, diff, floor, isnan, nanmax, nanmedian

    y = array(depth)
    bins = []
    bin_avg_sampling_freq = []

    if round_up:
        round_func = ceil
    else:
        round_func = floor

    d0 = 0
    d1 = chunk_depth
    last_freq = 0.5
    while d0 <= nanmax(depth):
        i = (y > d0) & (y < d1)

        bin_avg_sampling_freq += (nanmedian(abs(diff(y[i]))),)
        bin_freq = round_func(bin_avg_sampling_freq[-1] * 2) / 2
        if bin_freq == 0:
            bin_freq = 0.5
        elif isnan(bin_freq):
            bin_freq = last_freq
        bin_step = arange(d0, d1, bin_freq).tolist()
        bins += bin_step

        d0 = bin_step[-1] + bin_freq
        d1 = d0 + chunk_depth

        last_freq = bin_freq

    return array(bins), nanmedian(bin_avg_sampling_freq)


def grid_flat_dataarray(xda, bins=None):
    """
    Will grid an xr.DataArray if it contains coordinates correct metadata

    Parameters
    ----------
    xda : xr.DataArray
        flattened profile array with coordinates:
        (axis='T' | 'time' in name) and (axis='Z' | 'depth' in name)

    Returns a gridded dataset
    -------------------------
    """
    from .helpers import GliderToolsError

    has_requirements = 0
    for key in xda.coords:
        coord = xda[key]
        attrs = coord.attrs
        if ("Z" in attrs.get("axis", "").upper()) | ("depth" in coord.name):
            depth = coord
            has_requirements += 1
        if "dives" in coord.name:
            dives = coord
            has_requirements += 1
    if has_requirements == 2:
        x = dives
        y = depth
        z = xda

        gridded = grid_data(x, y, z, bins=bins, return_xarray=True, verbose=False)
        return gridded
    else:
        raise GliderToolsError(
            "The array coordinates do not contain axis info for gridding"
        )


try:
    import pykrige as pk
except ImportError:
    pk = None


def variogram(
    variable,
    horz,
    vert,
    dives,
    mask=None,
    xy_ratio=1,
    max_points=5000,
    ax=True,
):
    """
    Find the interpolation parameters and x and y scaling of the
    horizontal and vertical coordinate paramters for objective
    interpolation (Kriging).

    The range of the variogram will automatically be scaled to 1 and the x
    and y length scales will be given in the output. This can be used in
    the gt.mapping.interp_obj function.

    Parameters
    ----------
    variable : array-like
        target variable for interpolation as a flat array
    horz : array-like
        the horizontal coordinate variable as a flat array. Can be time in
        np.datetime64 format, or any other float value
    vert : array-like
        the vertical coordinate variable as a flat array. Usually depth,
        but can be pessure or any other variable.
    dives : array-like
        the dive numbers as a flat array
    mask : array-like
        a boolean array that can be used to retain certain regions,
        e.g. depth < 250
    xy_ratio : float
        determines the anisotropy of the coordinate variables. The value
        can be changed iteritively to improve the shape of the semivariance
        curve. The range of the variogram will automatically be scaled to
        1. The x and y length scales are given in the output.
    max_points : int
        maximum number of points that will be used in the variogram. The
        function selects a subset of dives rather than random points to be
        consistent in the vertical. Increasing this number will increase
        the accuracy of the length scales.
    ax : bool or mpl.Axes
        If True, will automatically create an axes, if an axes object is
        given, returns the plot in those axes.

    Returns
    -------
    variogram_params : dict
        a dictionary containing the information required by the
        gt.mapping.interp_obj function.
    plot : axes
        a axes object containing the plot of the semivariogram

    Example
    -------
    >>> gt.mapping.variogram(var, time, depth, dives, mask=depth<350,
                             xy_ratio=0.5, max_points=6000)

    """

    if pk is None:
        import warnings

        message = (
            "PyKrige is not installed. To enable the variogram function please "
            "run `pip install pykrige`. Variograms are required for sensible "
            "2D interpolation."
        )
        warnings.warn(message, category=GliderToolsWarning)

    def make_subset_index(dives, max_points):
        idx_points = (np.arange(dives.size) % 2).astype(bool)
        dives_half = dives[idx_points]

        dive_inds, counts = np.unique(dives_half, return_counts=True)
        average_count = counts.mean()
        n_dives = max_points // average_count
        step = int(dive_inds.size // n_dives)
        step = step if step > 0 else 1

        subsampled_dives = dive_inds[::step]
        idx_dives = np.any([dives == d for d in subsampled_dives], axis=0)

        idx = idx_dives & idx_points
        idx = idx & (idx.astype(int).cumsum() < max_points)

        return idx

    def plot_variogram(model, ax, n_dives):

        params = model.variogram_model_parameters
        psill, range, nugget = params
        sill = psill + nugget

        x = np.r_[0, model.lags]
        y = np.r_[nugget, model.semivariance]
        yhat = model.variogram_function(params, x)

        has_dots = np.any(
            [
                getattr(child, "get_marker", lambda: None)()
                for child in ax.get_children()
            ]
        )

        if not has_dots:
            ax.plot(x, y, ".k", label="Semivariance")

        ax.plot(x, yhat, "-", lw=4, label="Gaussian model")[0]
        ax.hlines(
            sill,
            0,
            params[1],
            color="orange",
            linestyle="--",
            linewidth=2.5,
            label="Sill ({:.2g})".format(sill),
        )
        ax.hlines(
            nugget,
            0,
            params[1],
            color="red",
            linestyle="--",
            linewidth=2.5,
            label="Nugget ({:.2g})".format(nugget),
        )
        ax.vlines(
            range,
            0,
            sill,
            color="#CCCCCC",
            linestyle="-",
            zorder=0,
            label="Range (1.0)",
        )

        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.legend()

        # generate info text
        # text = '${}$'.format(model.variogram_model.capitalize())
        # text += '\n sill: {:.2g}'.format(psill + nugget)
        # text += '\n nugget: {:.2g}'.format(nugget)

        # info text position
        # p = ax.get_position()
        # xt = p.x1 - 0.04 * (p.x1 - p.x0)
        # yt = p.y0 + 0.04 * (p.y1 - p.y0)
        # ax.get_figure().text(xt, yt, text, ha='right', va='bottom')

    # making all inputs arrays
    mask = np.ones_like(dives).astype(bool) if mask is None else mask
    variable, horz, vert, dives = [
        np.array(a)[mask] for a in [variable, horz, vert, dives]
    ]

    # creating a mask that selects dives rather than random points
    # then remove the nans and the points that are masked out
    subs = make_subset_index(dives, max_points) & ~np.isnan(variable)

    z = variable[subs]
    y = vert[subs]
    if np.issubdtype(horz.dtype, np.datetime64):
        x = horz[subs].astype("datetime64[s]").astype(float) / 3600
    else:
        x = horz[subs].astype(float)

    # creating initial scaling with anisotropy
    xlen = 1 / xy_ratio
    ylen = 1
    # and finding inital estiamte of range
    props = dict(weight=True, nlags=40, variogram_model="gaussian")
    gauss = pk.OrdinaryKriging(x / xlen, y / ylen, z, **props)

    # scale initial scaling by range
    xlen *= gauss.variogram_model_parameters[1]
    ylen *= gauss.variogram_model_parameters[1]
    # calculate new variogram
    gauss = pk.OrdinaryKriging(x / xlen, y / ylen, z, **props)

    # making plots
    if (ax is not None) or (ax is not False):
        if not isinstance(ax, plt.Axes):
            fig, ax = plt.subplots(figsize=[6, 4], dpi=100)

        n_dives = np.unique(dives[subs]).size
        t_dives = np.unique(dives).size
        plot_variogram(gauss, ax, n_dives)
        ax.set_xlabel("Scaled lag (x = {:.0f}; y = {:.0f})".format(xlen, ylen))
        ax.set_ylabel("Semivariance\n(using {} of {} dives)".format(n_dives, t_dives))
    else:
        ax = None

    # creating a dict of model parameters used for interpolation
    full_mask = np.array(mask.copy()) * False
    full_mask[mask] = subs
    output = dict(
        partial_sill=gauss.variogram_model_parameters[0],
        nugget=gauss.variogram_model_parameters[2],
        lenscale_x=xlen,
        lenscale_y=ylen,
        mask=full_mask,
    )

    return output, ax

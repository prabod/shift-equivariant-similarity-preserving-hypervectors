"""
.. Copyright (c) 2022 Prabod Rathnayaka
   license http://opensource.org/licenses/MIT
Common - generic functions
==================================================
"""
import time

import numpy as np
import logging


def extract_local_binary_pattern(patch, mask_idx, pad=0):
    """
    Extract Local Binary Features from a given image patch and mask. Mask should be in the form of non-zero index
    values of rows and columns. The image patch and mask should have the same dimensions.

    Parameters
    ----------
    patch : ndarray
        k input image patches of size nxn. i.e : ndarray of shape (k, n, n)
    mask_idx : Tuple(ndarray,ndarray)
        Non-zero element indexes of the mask. (row_indexes,col_indexes)
    pad : int
        Padding value for np.nan values. Default = 0

    Returns
    -------
    out : ndarray
        Outputs a ndarray of shape (k,). Each element is the decimal representation of the output after applying mask.
    """

    rows, cols = mask_idx
    assert max(patch.shape[1:]) > max([rows.max(), cols.max()]), "Patch and Mask should be of the same size or smaller."
    bits = np.nan_to_num(patch, nan=pad)[:, rows, cols]
    median_bit = (np.median(bits, axis=-1, keepdims=True)).astype(np.uint8)
    bits -= median_bit
    bits = np.clip(bits, 0, 1)
    return bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))


def get_lbp_features(patch, mask_idx):
    """
    Helper function to extract Local Binary Features for all the patches in a given image and mask. Mask should be in
    the form of non-zero index values of rows and columns. The image patch and mask should have the same dimensions.

    Parameters
    ----------
    patch : ndarray
        k input images of size mxm with patches of size nxn for each pixel. i.e : ndarray of shape (k, m, m, n, n)
    mask_idx : Tuple(ndarray,ndarray)
        Non-zero element indexes of the mask. (row_indexes,col_indexes)

    Returns
    -------
    out : ndarray
        Outputs a ndarray of shape (k, m, m). Each element is the decimal representation of the output after applying
        mask.
    """
    row, col = mask_idx
    lbp_features = np.zeros(patch.shape[:3])
    for e in range(patch.shape[1]):
        for f in range(patch.shape[2]):
            lbp_features[:, e, f] = extract_local_binary_pattern(patch[:, e, f, :, :], [row, col])
    return np.array(lbp_features).reshape(patch.shape[:3])


def cycle_permutation(n, seed=101):
    """
    Generate a permutation for a vector of size n.
    Parameters
    ----------
    n : int
        Dimension of the vector
    seed : int
        Random seed

    Returns
    -------
    out: ndarray
        Ndarray of size n
    """
    np.random.seed(seed)
    a = np.arange(n)
    for i in range(n - 1):
        j = np.random.randint(i + 1, n)
        a[[j, i]] = a[[i, j]]
    return a


def shifts_by_permutations_inverse_fast(in_cv_array, perm_y, perm_x, perm_y1, perm_x1):
    in_cv_array = perm_y[in_cv_array.astype(int)]
    in_cv_array = perm_x[in_cv_array.astype(int)]
    in_cv_array = perm_y1[in_cv_array.astype(int)]
    out_cv_array = perm_x1[in_cv_array.astype(int)]
    return out_cv_array


def prepare_permutations(ysz, xsz, Dc, Dc_ratio, N, seed, ysz1, xsz1, Dc2, Dc2_ratio, cyc):
    DcX = round(Dc * Dc_ratio)
    DcX2 = round(Dc2 * Dc2_ratio)

    perm_y0 = cycle_permutation(N, seed)
    invperm_y0 = np.arange(N)
    invperm_y0[perm_y0] = np.arange(N)
    no_perm = np.arange(N).T

    perm_y = [np.copy(perm_y0)]
    perm_compos_col = np.copy(perm_y0)

    for k in range(1, ysz + Dc - 1):
        perm_compos_col = np.copy(perm_y0[perm_compos_col])
        perm_y.append(np.copy(perm_compos_col))
    perm_y = np.array(perm_y).T

    perm_y_inv = [np.copy(invperm_y0)]
    perm_compos_col_inv = np.copy(invperm_y0)
    for k in range(1, ysz + Dc - 1):
        perm_compos_col_inv = np.copy(invperm_y0[perm_compos_col_inv])
        perm_y_inv.append(np.copy(perm_compos_col_inv))
    perm_y_inv = np.fliplr(np.array(perm_y_inv).T)
    perm_y_all = np.concatenate([perm_y_inv, np.expand_dims(no_perm, -1), perm_y], axis=-1)

    perm_x = [np.copy(perm_compos_col)]
    perm_compos_col1 = np.copy(perm_compos_col)

    for k in range(1, xsz + DcX - 1):
        perm_compos_col1 = np.copy(perm_compos_col[perm_compos_col1])
        perm_x.append(np.copy(perm_compos_col[perm_x[k - 1]]))
    perm_x = np.array(perm_x).T

    perm_x_inv = []
    perm_x_inv.append(np.copy(perm_compos_col_inv))
    perm_compos_col_inv1 = np.copy(perm_compos_col_inv)
    for k in range(xsz + DcX - 1):
        perm_compos_col_inv1 = np.copy(perm_compos_col_inv[perm_compos_col_inv1])
        perm_x_inv.append(np.copy(perm_compos_col_inv[perm_x_inv[k - 1]]))

    perm_x_inv = np.fliplr(np.array(perm_x_inv).T)
    perm_x_all = np.concatenate([perm_x_inv, np.expand_dims(no_perm, -1), perm_x], axis=-1)

    perm_y1 = [np.copy(perm_compos_col1)]
    perm_y_inv1 = [np.copy(perm_compos_col_inv1)]

    perm_compos_col2 = np.copy(perm_compos_col1)
    perm_compos_col_inv2 = np.copy(perm_compos_col_inv1)

    if ysz > 1:
        for k in range(1, ysz1 + Dc2 - 1):
            perm_compos_col2 = np.copy(perm_compos_col1[perm_compos_col2])
            perm_y1.append(np.copy(perm_compos_col1[perm_y1[k - 1]]))

        perm_y1 = np.array(perm_y1).T

        for k in range(1, ysz1 + Dc2 - 1):
            perm_compos_col_inv2 = np.copy(perm_compos_col_inv1[perm_compos_col_inv2])
            perm_y_inv1.append(np.copy(perm_compos_col_inv1[perm_y_inv1[k - 1]]))

        perm_y_inv1 = np.fliplr(np.array(perm_y_inv1).T)
        perm_y_all1 = np.concatenate([perm_y_inv1, np.expand_dims(no_perm, -1), perm_y1], axis=-1)

    perm_x1 = [np.copy(perm_compos_col2)]
    perm_x_inv1 = [np.copy(perm_compos_col_inv2)]

    perm_compos_col3 = np.copy(perm_compos_col2)
    perm_compos_col_inv3 = np.copy(perm_compos_col_inv2)
    if xsz > 1:
        for k in range(1, xsz1 + DcX2 - 1):
            perm_compos_col3 = np.copy(perm_compos_col2[perm_compos_col3])
            perm_x1.append(np.copy(perm_compos_col2[perm_x1[k - 1]]))

        perm_x1 = np.array(perm_x1).T

        for k in range(1, xsz1 + DcX2 - 1):
            perm_compos_col_inv3 = np.copy(perm_compos_col_inv2[perm_compos_col_inv3])
            perm_x_inv1.append(np.copy(perm_compos_col_inv2[perm_x_inv1[k - 1]]))

        perm_x_inv1 = np.fliplr(np.array(perm_x_inv1).T)
        perm_x_all1 = np.concatenate([perm_x_inv1, np.expand_dims(no_perm, -1), perm_x1], axis=-1)

    return perm_y_all, perm_x_all, perm_y_all1, perm_x_all1


def make_codevector_matrix(y_dim, x_dim, dc, dc_ratio, n, y2_dim, x2_dim, dc2, dc2_ratio, p, cyclic, seed=101):
    """
    Create a codevector matrix given x, y, x1, y1 dimensions.
    Parameters
    ----------
    y_dim : int
        y dimension
    x_dim : int
        x dimension
    dc : int
    dc_ratio : int
    n : int
        codevector dimension
    y2_dim :int
        y2 dimension
    x2_dim :int
        x2 dimension
    dc2 :int
    dc2_ratio : int
    p :int
    cyclic :int
        apply cyclic permutation
    seed : int
        Random seed. default = 101

    Returns
    -------
    out: ndarray
        Codevector matrix of the shape (y_dim, x_dim, y2_dim, x2_dim, dc*dc*dc_ratio*dc2*dc2*dc2*ratio*p)
    """
    np.random.seed(seed)
    dc_x = round(dc * dc_ratio)
    dc_x2 = round(dc2 * dc2_ratio)
    Np = round(p)
    num = 0
    t1 = time.time()
    perm_all = prepare_permutations(y_dim, x_dim, dc, dc_ratio, n, seed, y2_dim, x2_dim, dc2, dc2_ratio, cyclic)
    t2 = time.time()
    logging.info("Preparing permutations took %f seconds" % (t2 - t1))
    perm_y_all = perm_all[0]
    perm_x_all = perm_all[1]
    perm_y_all1 = perm_all[2]
    perm_x_all1 = perm_all[3]

    t1 = time.time()
    base_permutation = np.random.permutation(n)
    base_vec = base_permutation[:Np]
    in_cv1 = []

    if cyclic == 1:
        perm_y0 = cycle_permutation(n, seed)
    if cyclic == 0:
        perm_y0 = np.random.permutation(n)

    base_vec = np.copy(perm_y0[base_vec])

    for jj1 in range(dc_x2):
        perm_x1 = perm_x_all1[:, jj1 + x2_dim + dc_x2 - 1]

        for ii1 in range(dc2):
            perm_y1 = perm_y_all1[:, ii1 + y2_dim + dc2 - 1]

            for jj in range(dc_x):
                perm_x = perm_x_all[:, jj + x_dim + dc_x - 1]

                for ii in range(dc):
                    perm_y = perm_y_all[:, ii + y_dim + dc - 1]

                    in_cv1.append(shifts_by_permutations_inverse_fast(base_vec, perm_y, perm_x, perm_y1, perm_x1))
                    num = num + Np
    in_cv1 = np.array(in_cv1).flatten()
    codevector_matrix = np.zeros((y_dim, x_dim, y2_dim, x2_dim, dc * dc_x * dc2 * dc_x2 * Np), dtype=np.uint32)

    for jj1 in range(x2_dim):
        perm_x1 = perm_x_all1[:, jj1 + x2_dim + dc_x2 - 1]
        for ii1 in range(y2_dim):
            perm_y1 = perm_y_all1[:, ii1 + y2_dim + dc2 - 1]
            for jj in range(x_dim):
                perm_x = perm_x_all[:, jj + x_dim + dc_x - 1]
                for ii in range(y_dim):
                    perm_y = perm_y_all[:, ii + y_dim + dc - 1]
                    codevector_matrix[ii, jj, ii1, jj1, :] = shifts_by_permutations_inverse_fast(np.copy(in_cv1),
                                                                                                 perm_y,
                                                                                                 perm_x,
                                                                                                 perm_y1, perm_x1)
    t2 = time.time()
    logging.info("Preparing codevector matrix took %f seconds" % (t2 - t1))
    return codevector_matrix


def make_sparsevector(nodes, nodes2, codevector_matrix, num1_limit, m=0):
    """
    Create a sparse vector given LBP features of a image
    Parameters
    ----------
    nodes : ndarray
        Indexes of non-zero features
    nodes2 : ndarray
        Values of non-zero features
    codevector_matrix : ndarray
        Codevector Matrix
    num1_limit : int
        Upper bound of active bits in the sparse vector
    m : int
        row index of the sparse vector. default = 0

    Returns
    -------
    out: ndarray
        rows and cols of non-zero elements of the sparse vector

    """
    if nodes.shape != nodes2.shape:
        return
    out_codevector = np.zeros((nodes2.shape[1], codevector_matrix.shape[-1]))
    for i in range(nodes2.shape[1]):
        y = nodes[0, i]
        x = nodes[1, i]
        y2 = nodes2[1, i]
        x2 = nodes2[0, i]

        out_codevector[i, :] = codevector_matrix[y, x, int(y2), int(x2)]
    unique_index = np.unique(out_codevector.flatten())
    if num1_limit:
        num_non_zero = np.nonzero(unique_index)[0].shape[0]
        if num_non_zero > num1_limit:
            unique_index = unique_index[:num1_limit]
    return [m] * unique_index.shape[0], unique_index.tolist()

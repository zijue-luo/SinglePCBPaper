"""
Group peaks by overlap on each side (left/right) of a resonance.
Overlap: |mu1 - mu2| <= K * sqrt(sigma1^2 + sigma2^2). Default K=1.
Uses Union-Find for transitive closure: any two peaks that overlap (directly or
via a chain) end up in the same group. Prevents adjacent groups with overlapping ranges.
"""
import numpy as np


def _overlaps(mu1, sigma1, mu2, sigma2, K=1.0):
    """True if the two Gaussians overlap under criterion |mu1-mu2| <= K*sqrt(sigma1^2+sigma2^2)."""
    d = abs(float(mu1) - float(mu2))
    thresh = K * np.sqrt(float(sigma1) ** 2 + float(sigma2) ** 2)
    return d <= thresh


def _union_find_parent(parent, i):
    if parent[i] != i:
        parent[i] = _union_find_parent(parent, parent[i])
    return parent[i]


def _union_find_merge(parent, i, j):
    pi, pj = _union_find_parent(parent, i), _union_find_parent(parent, j)
    if pi != pj:
        parent[pi] = pj


def group_overlapping_peaks(peaks, K=1.0):
    """
    Group peaks that overlap (by mu/sigma criterion). Uses Union-Find for full
    transitive closure: if A overlaps B and B overlaps C, all three are in one group.
    This avoids adjacent groups with overlapping [mu±sigma] ranges.

    Parameters
    ----------
    peaks : list of (amp, mu, sigma) or list of dicts with keys 'mu', 'sigma'
        All peaks on one side (e.g. left or right).
    K : float
        Overlap threshold factor: overlap if |mu1-mu2| <= K*sqrt(sigma1^2+sigma2^2).

    Returns
    -------
    groups : list of list of int
        Each element is a group: list of **original** indices into `peaks`.
    """
    if not peaks:
        return []

    if isinstance(peaks[0], dict):
        list_peaks = [(p.get("amp", 0), float(p["mu"]), float(p["sigma"])) for p in peaks]
    else:
        list_peaks = [(float(p[0]), float(p[1]), float(p[2])) for p in peaks]

    n = len(list_peaks)
    parent = list(range(n))

    for i in range(n):
        mu_i, sigma_i = list_peaks[i][1], list_peaks[i][2]
        for j in range(i + 1, n):
            mu_j, sigma_j = list_peaks[j][1], list_peaks[j][2]
            if _overlaps(mu_i, sigma_i, mu_j, sigma_j, K):
                _union_find_merge(parent, i, j)

    # Collect groups by root
    root_to_indices = {}
    for i in range(n):
        r = _union_find_parent(parent, i)
        root_to_indices.setdefault(r, []).append(i)

    groups_original = [sorted(idxs) for idxs in root_to_indices.values()]
    # Sort groups by min(mu) so order is stable (low to high)
    groups_original.sort(key=lambda g: min(list_peaks[i][1] for i in g))
    return groups_original

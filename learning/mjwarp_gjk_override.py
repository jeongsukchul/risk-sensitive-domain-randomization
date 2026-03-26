"""Repo-local override for vendored MJWarp GJK.

This mirrors the upstream PR #1239 convergence fix while keeping the override in
the repository so it can be wired in at runtime.
"""

from __future__ import annotations

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import _almost_equal
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import _linear_combine
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import _subdistance
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import FLOAT_MAX
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import Geom
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import GJKResult
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import mat43
from mujoco.mjx.third_party.mujoco_warp._src.collision_gjk import support


@wp.func
def gjk(
  # In:
  tolerance: float,
  gjk_iterations: int,
  geom1: Geom,
  geom2: Geom,
  x1_0: wp.vec3,
  x2_0: wp.vec3,
  geomtype1: int,
  geomtype2: int,
  cutoff: float,
  is_discrete: bool,
) -> GJKResult:
  """Find distance within a tolerance between two geoms."""
  cutoff2 = cutoff * cutoff
  simplex = mat43()
  simplex1 = mat43()
  simplex2 = mat43()
  simplex_index1 = wp.vec4i()
  simplex_index2 = wp.vec4i()
  n = int(0)
  cnt = int(1)
  coordinates = wp.vec4()  # barycentric coordinates
  tol2 = tolerance * tolerance
  epsilon = wp.where(is_discrete, 0.0, 0.5 * tol2)

  # set initial guess
  x_k = x1_0 - x2_0
  xnorm_old = FLOAT_MAX

  for _ in range(gjk_iterations):
    xnorm = wp.dot(x_k, x_k)
    # Upstream PR #1239: stop if we are within tolerance or no longer improving.
    if xnorm < tol2 or wp.abs(xnorm_old - xnorm) < tol2:
      break
    xnorm_old = xnorm
    dir_neg = x_k / wp.sqrt(xnorm)

    # compute kth support point in geom1
    sp = support(geom1, geomtype1, -dir_neg)
    simplex1[n] = sp.point
    geom1.index = sp.cached_index
    simplex_index1[n] = sp.vertex_index

    # compute kth support point in geom2
    sp = support(geom2, geomtype2, dir_neg)
    simplex2[n] = sp.point
    geom2.index = sp.cached_index
    simplex_index2[n] = sp.vertex_index

    # compute the kth support point
    simplex[n] = simplex1[n] - simplex2[n]

    if cutoff == 0.0:
      if wp.dot(x_k, simplex[n]) > 0.0:
        result = GJKResult()
        result.dim = 0
        result.dist = FLOAT_MAX
        return result
    elif cutoff < FLOAT_MAX:
      vs = wp.dot(x_k, simplex[n])
      if wp.dot(x_k, simplex[n]) > 0.0 and (vs * vs / xnorm) >= cutoff2:
        result = GJKResult()
        result.dim = 0
        result.dist = FLOAT_MAX
        return result

    # stopping criteria using the Frank-Wolfe duality gap given by
    #  |f(x_k) - f(x_min)|^2 <= < grad f(x_k), (x_k - simplex[n]) >
    if wp.dot(x_k, x_k - simplex[n]) < epsilon:
      break

    # run the distance subalgorithm to compute the barycentric coordinates
    # of the closest point to the origin in the simplex
    coordinates = _subdistance(n + 1, simplex)

    # remove vertices from the simplex no longer needed
    n = int(0)
    for i in range(4):
      if coordinates[i] == 0.0:
        continue

      simplex[n] = simplex[i]
      simplex1[n] = simplex1[i]
      simplex2[n] = simplex2[i]
      simplex_index1[n] = simplex_index1[i]
      simplex_index2[n] = simplex_index2[i]
      coordinates[n] = coordinates[i]
      n += int(1)

    # SHOULD NOT OCCUR
    if n < 1:
      break

    # get the next iteration of x_k
    x_next = _linear_combine(n, coordinates, simplex)

    # x_k has converged to minimum
    if _almost_equal(x_next, x_k):
      break

    # copy next iteration into x_k
    x_k = x_next

    # we have a tetrahedron containing the origin so return early
    if n == 4:
      break

    cnt += 1

  if cnt == gjk_iterations:
    wp.printf("Warning: opt.ccd_iterations, currently set to %d, needs to be increased.\n", gjk_iterations)

  result = GJKResult()

  # compute the approximate witness points
  # if n is zero, then there was an immediate return meaning the initial points
  # are the witness points
  result.x1 = wp.where(n == 0, x1_0, _linear_combine(n, coordinates, simplex1))
  result.x2 = wp.where(n == 0, x2_0, _linear_combine(n, coordinates, simplex2))
  result.dist = wp.norm_l2(x_k)

  result.dim = n
  result.simplex1 = simplex1
  result.simplex2 = simplex2
  result.simplex_index1 = simplex_index1
  result.simplex_index2 = simplex_index2
  result.simplex = simplex
  return result

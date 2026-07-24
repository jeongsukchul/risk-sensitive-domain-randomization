import unittest

import jax
import jax.numpy as jnp
import numpy as np

from learning.agents.sampler_ppo.dual_optimization import (
    beta_from_dual,
    clipped_kl_violation,
    dual_from_beta,
    estimate_kl_to_uniform,
    exponential_moving_average,
    projected_direct_beta_update,
    projected_dual_ascent,
)


class DualOptimizationTest(unittest.TestCase):

  def test_beta_dual_round_trip(self):
    beta = jnp.asarray(-20.0)
    dual_lambda = dual_from_beta(beta)

    np.testing.assert_allclose(dual_lambda, 0.05)
    np.testing.assert_allclose(beta_from_dual(dual_lambda), beta)

  def test_uniform_kl_estimate_is_zero(self):
    low = jnp.asarray([-2.0, 1.0])
    high = jnp.asarray([2.0, 3.0])
    log_uniform = -jnp.sum(jnp.log(high - low))
    log_q = jnp.full((128,), log_uniform)

    np.testing.assert_allclose(
        estimate_kl_to_uniform(log_q, low, high), 0.0, atol=1e-6
    )

  def test_kl_estimate_uses_uniform_reference_volume(self):
    low = jnp.asarray([-2.0, 0.0])
    high = jnp.asarray([2.0, 2.0])  # Reference volume is 8.
    log_q = jnp.full((128,), -jnp.log(2.0))  # q has constant density 1/2.

    np.testing.assert_allclose(
        estimate_kl_to_uniform(log_q, low, high),
        jnp.log(4.0),
        atol=1e-6,
    )

  def test_exponential_moving_average(self):
    updated = exponential_moving_average(
        previous_value=0.1,
        current_value=1.1,
        decay=0.9,
    )
    np.testing.assert_allclose(updated, 0.2, atol=1e-6)

  def test_kl_violation_is_clipped_symmetrically(self):
    positive = clipped_kl_violation(1.0, 0.1, 0.25)
    negative = clipped_kl_violation(0.0, 1.0, 0.25)

    np.testing.assert_allclose(positive, 0.25)
    np.testing.assert_allclose(negative, -0.25)

  def test_dual_ascent_direction_and_projection(self):
    increased, violation = projected_dual_ascent(
        dual_lambda=0.5,
        estimated_kl=0.3,
        kl_radius=0.1,
        learning_rate=0.5,
        min_dual=0.01,
        max_dual=1.0,
    )
    decreased, _ = projected_dual_ascent(
        dual_lambda=0.5,
        estimated_kl=0.0,
        kl_radius=0.1,
        learning_rate=0.5,
        min_dual=0.01,
        max_dual=1.0,
    )
    clipped, _ = projected_dual_ascent(
        dual_lambda=0.99,
        estimated_kl=10.0,
        kl_radius=0.1,
        learning_rate=1.0,
        min_dual=0.01,
        max_dual=1.0,
    )

    np.testing.assert_allclose(violation, 0.2)
    np.testing.assert_allclose(increased, 0.6)
    np.testing.assert_allclose(decreased, 0.45)
    np.testing.assert_allclose(clipped, 1.0)

  def test_dual_ascent_uses_clipped_violation(self):
    updated, violation = projected_dual_ascent(
        dual_lambda=0.5,
        estimated_kl=1.0,
        kl_radius=0.1,
        learning_rate=1.0,
        min_dual=0.01,
        max_dual=10.0,
        max_abs_violation=0.25,
    )

    np.testing.assert_allclose(violation, 0.25)
    np.testing.assert_allclose(updated, 0.75)

  def test_direct_beta_update_is_naive_and_projected(self):
    updated, violation = projected_direct_beta_update(
        beta=-2.0,
        estimated_kl=0.3,
        kl_radius=0.1,
        learning_rate=0.5,
        min_beta=-10.0,
        max_beta=-0.01,
    )
    clipped, _ = projected_direct_beta_update(
        beta=-0.02,
        estimated_kl=10.0,
        kl_radius=0.1,
        learning_rate=1.0,
        min_beta=-10.0,
        max_beta=-0.01,
    )

    np.testing.assert_allclose(violation, 0.2)
    np.testing.assert_allclose(updated, -1.9)
    np.testing.assert_allclose(clipped, -0.01)

  def test_direct_beta_update_uses_clipped_violation(self):
    updated, violation = projected_direct_beta_update(
        beta=-1.0,
        estimated_kl=1.0,
        kl_radius=0.1,
        learning_rate=1.0,
        min_beta=-10.0,
        max_beta=-0.01,
        max_abs_violation=0.25,
    )

    np.testing.assert_allclose(violation, 0.25)
    np.testing.assert_allclose(updated, -0.75)

  def test_helpers_are_jittable(self):
    update = jax.jit(
        lambda dual_lambda, estimated_kl: projected_dual_ascent(
            dual_lambda,
            estimated_kl,
            kl_radius=0.2,
            learning_rate=0.1,
            min_dual=1e-4,
            max_dual=10.0,
        )
    )

    updated_dual, violation = update(jnp.asarray(0.5), jnp.asarray(0.4))
    np.testing.assert_allclose(updated_dual, 0.52)
    np.testing.assert_allclose(violation, 0.2)

    beta_update = jax.jit(
        lambda beta, estimated_kl: projected_direct_beta_update(
            beta,
            estimated_kl,
            kl_radius=0.2,
            learning_rate=0.1,
            min_beta=-100.0,
            max_beta=-1e-4,
        )
    )
    updated_beta, violation = beta_update(
        jnp.asarray(-1.0), jnp.asarray(0.4)
    )
    np.testing.assert_allclose(updated_beta, -0.98)
    np.testing.assert_allclose(violation, 0.2)


if __name__ == "__main__":
  unittest.main()

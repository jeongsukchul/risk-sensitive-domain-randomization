import unittest

import jax
import jax.numpy as jnp
import numpy as np

from learning.agents.sampler_ppo.empirical_metrics import (
    compare_empirical_target_and_sampler,
    empirical_target_noise_metrics,
)


class EmpiricalMetricsTest(unittest.TestCase):

  def test_identical_rollout_targets_have_zero_noise(self):
    per_rollout_returns = jnp.tile(
        jnp.asarray([[0.0, 1.0, 2.0]]), (10, 1)
    )

    metrics = empirical_target_noise_metrics(
        per_rollout_returns, beta=-2.0
    )

    for name in (
        "target_logit_se_mean",
        "target_logit_se_p95",
        "target_logit_se_max",
        "target_split_reverse_kl",
        "target_split_forward_kl",
        "target_split_js_divergence",
        "target_split_total_variation",
    ):
      np.testing.assert_allclose(metrics[name], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        metrics["target_split_overlap"], 1.0, atol=1e-6
    )

  def test_split_target_metric_detects_rollout_noise(self):
    per_rollout_returns = jnp.asarray([
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    metrics = empirical_target_noise_metrics(
        per_rollout_returns, beta=-4.0
    )

    self.assertGreater(
        float(metrics["target_split_total_variation"]), 0.9
    )
    self.assertLess(float(metrics["target_split_overlap"]), 0.1)
    self.assertGreater(float(metrics["target_logit_se_max"]), 0.0)

  def test_identical_distributions_have_zero_divergence(self):
    returns = jnp.asarray([-1.0, 0.0, 1.0])
    beta = -2.0
    sampler_logits = beta * returns + 17.0

    metrics, log_target, log_sampler, _ = (
        compare_empirical_target_and_sampler(
            returns, beta, sampler_logits
        )
    )

    np.testing.assert_allclose(log_target, log_sampler, atol=1e-6)
    for name in (
        "reverse_kl_q_to_target",
        "forward_kl_target_to_q",
        "js_divergence",
        "total_variation",
        "hellinger_distance",
        "absolute_expected_return_gap",
    ):
      np.testing.assert_allclose(metrics[name], 0.0, atol=1e-6)
    np.testing.assert_allclose(metrics["overlap"], 1.0, atol=1e-6)

  def test_negative_beta_favors_low_returns(self):
    returns = jnp.asarray([-2.0, 0.0, 2.0])

    _, log_target, _, center = compare_empirical_target_and_sampler(
        returns,
        beta=-3.0,
        sampler_log_density=jnp.zeros_like(returns),
    )

    self.assertEqual(float(center), -2.0)
    target_mass = jnp.exp(log_target)
    self.assertGreater(float(target_mass[0]), float(target_mass[1]))
    self.assertGreater(float(target_mass[1]), float(target_mass[2]))

  def test_uniform_target_has_zero_reverse_kl_to_uniform(self):
    returns = jnp.asarray([-3.0, -1.0, 2.0, 7.0])

    metrics, _, _, _ = compare_empirical_target_and_sampler(
        returns,
        beta=0.0,
        sampler_log_density=jnp.zeros_like(returns),
    )

    np.testing.assert_allclose(
        metrics["target_reverse_kl_to_uniform"], 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["sampler_reverse_kl_to_uniform"], 0.0, atol=1e-6
    )

  def test_metrics_match_direct_probability_calculation(self):
    returns = jnp.asarray([0.0, 1.0])
    beta = 1.0
    sampler_mass = jnp.asarray([0.75, 0.25])

    metrics, log_target, _, _ = compare_empirical_target_and_sampler(
        returns,
        beta,
        jnp.log(sampler_mass),
    )
    target_mass = jax.nn.softmax(beta * returns)
    expected_reverse_kl = jnp.sum(
        sampler_mass * (jnp.log(sampler_mass) - jnp.log(target_mass))
    )
    expected_forward_kl = jnp.sum(
        target_mass * (jnp.log(target_mass) - jnp.log(sampler_mass))
    )

    np.testing.assert_allclose(jnp.exp(log_target), target_mass, atol=1e-6)
    np.testing.assert_allclose(
        metrics["reverse_kl_q_to_target"], expected_reverse_kl, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["forward_kl_target_to_q"], expected_forward_kl, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["target_log_normalizer"],
        jax.scipy.special.logsumexp(beta * returns),
        atol=1e-6,
    )
    expected_target_kl_to_uniform = jnp.sum(
        target_mass * (jnp.log(target_mass) + jnp.log(2.0))
    )
    expected_sampler_kl_to_uniform = jnp.sum(
        sampler_mass * (jnp.log(sampler_mass) + jnp.log(2.0))
    )
    np.testing.assert_allclose(
        metrics["target_reverse_kl_to_uniform"],
        expected_target_kl_to_uniform,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["sampler_reverse_kl_to_uniform"],
        expected_sampler_kl_to_uniform,
        atol=1e-6,
    )

  def test_comparison_is_jittable(self):
    compare = jax.jit(compare_empirical_target_and_sampler)
    metrics, log_target, log_sampler, _ = compare(
        jnp.asarray([0.0, 1.0, 2.0]),
        -1.0,
        jnp.asarray([-0.2, -0.5, -1.0]),
    )

    self.assertTrue(bool(jnp.all(jnp.isfinite(log_target))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(log_sampler))))
    self.assertTrue(bool(jnp.isfinite(metrics["js_divergence"])))


if __name__ == "__main__":
  unittest.main()

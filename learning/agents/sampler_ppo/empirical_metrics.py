"""Discrete-grid metrics for comparing an empirical target and a sampler."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def compare_empirical_target_and_sampler(
    estimated_returns,
    beta,
    sampler_log_density,
) -> Tuple[Dict[str, jax.Array], jax.Array, jax.Array, jax.Array]:
  """Compares softmax(beta * return) with a sampler on an equal-area grid.

  The target centering constant is chosen so every target logit is non-positive:
  min(return) for negative beta and max(return) otherwise.  This constant has no
  effect after normalization.

  Args:
    estimated_returns: Estimated J(theta, xi_i) at each grid cell.
    beta: Current inverse temperature.
    sampler_log_density: Continuous sampler log density at each grid cell.

  Returns:
    A tuple of metrics, normalized target log masses, normalized sampler log
    masses, and the target centering constant.
  """
  estimated_returns = jnp.asarray(estimated_returns).reshape(-1)
  sampler_log_density = jnp.asarray(sampler_log_density).reshape(-1)
  beta = jnp.asarray(beta)

  center = jnp.where(
      beta < 0,
      jnp.min(estimated_returns),
      jnp.max(estimated_returns),
  )
  target_logits = beta * (estimated_returns - center)
  target_logsumexp_centered = jax.scipy.special.logsumexp(target_logits)
  log_target = jax.nn.log_softmax(target_logits)
  log_sampler = jax.nn.log_softmax(sampler_log_density)

  target_mass = jnp.exp(log_target)
  sampler_mass = jnp.exp(log_sampler)
  log_midpoint = jnp.logaddexp(log_target, log_sampler) - jnp.log(2.0)

  reverse_kl = jnp.sum(sampler_mass * (log_sampler - log_target))
  forward_kl = jnp.sum(target_mass * (log_target - log_sampler))
  js_divergence = 0.5 * (
      jnp.sum(target_mass * (log_target - log_midpoint))
      + jnp.sum(sampler_mass * (log_sampler - log_midpoint))
  )
  total_variation = 0.5 * jnp.sum(jnp.abs(target_mass - sampler_mass))
  hellinger = jnp.sqrt(
      0.5 * jnp.sum(
          jnp.square(jnp.sqrt(target_mass) - jnp.sqrt(sampler_mass))
      )
  )
  overlap = jnp.sum(jnp.minimum(target_mass, sampler_mass))
  num_cells = target_mass.shape[0]
  target_ess_fraction = 1.0 / (
      num_cells * jnp.sum(jnp.square(target_mass))
  )
  sampler_ess_fraction = 1.0 / (
      num_cells * jnp.sum(jnp.square(sampler_mass))
  )
  target_expected_return = jnp.sum(target_mass * estimated_returns)
  sampler_expected_return = jnp.sum(sampler_mass * estimated_returns)
  log_num_cells = jnp.log(jnp.asarray(num_cells, dtype=target_mass.dtype))
  target_reverse_kl_to_uniform = (
      log_num_cells + jnp.sum(target_mass * log_target)
  )
  sampler_reverse_kl_to_uniform = (
      log_num_cells + jnp.sum(sampler_mass * log_sampler)
  )

  metrics = {
      "reverse_kl_q_to_target": reverse_kl,
      "forward_kl_target_to_q": forward_kl,
      "js_divergence": js_divergence,
      "total_variation": total_variation,
      "hellinger_distance": hellinger,
      "overlap": overlap,
      "target_ess_fraction": target_ess_fraction,
      "sampler_ess_fraction": sampler_ess_fraction,
      "target_reverse_kl_to_uniform": target_reverse_kl_to_uniform,
      "sampler_reverse_kl_to_uniform": sampler_reverse_kl_to_uniform,
      "target_entropy": -jnp.sum(target_mass * log_target),
      "sampler_entropy": -jnp.sum(sampler_mass * log_sampler),
      "target_logsumexp_centered": target_logsumexp_centered,
      "target_log_normalizer": (
          beta * center + target_logsumexp_centered
      ),
      "target_expected_return": target_expected_return,
      "sampler_expected_return": sampler_expected_return,
      "absolute_expected_return_gap": jnp.abs(
          sampler_expected_return - target_expected_return
      ),
  }
  return metrics, log_target, log_sampler, center

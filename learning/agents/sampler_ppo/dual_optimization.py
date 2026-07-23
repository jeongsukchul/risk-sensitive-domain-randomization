"""Dual optimization utilities for fixed-KL-radius domain randomization."""

import jax.numpy as jnp


def dual_from_beta(beta):
  """Converts a negative inverse temperature beta to lambda = -1 / beta."""
  return -jnp.reciprocal(jnp.asarray(beta))


def beta_from_dual(dual_lambda):
  """Converts a positive dual variable lambda to beta = -1 / lambda."""
  return -jnp.reciprocal(jnp.asarray(dual_lambda))


def estimate_kl_to_uniform(log_q, low, high):
  """Monte Carlo estimate of KL(q || Uniform([low, high])).

  Args:
    log_q: Log density of q evaluated at samples drawn from q.
    low: Lower bounds of the uniform reference distribution.
    high: Upper bounds of the uniform reference distribution.

  Returns:
    The scalar estimate E_q[log q(x) - log p_uniform(x)].
  """
  log_volume = jnp.sum(jnp.log(jnp.asarray(high) - jnp.asarray(low)))
  return jnp.mean(jnp.asarray(log_q)) + log_volume


def projected_dual_ascent(
    dual_lambda,
    estimated_kl,
    kl_radius,
    learning_rate,
    min_dual,
    max_dual,
):
  """Performs lambda <- projection(lambda + lr * (KL - radius))."""
  violation = jnp.asarray(estimated_kl) - jnp.asarray(kl_radius)
  updated_dual = jnp.clip(
      jnp.asarray(dual_lambda) + jnp.asarray(learning_rate) * violation,
      jnp.asarray(min_dual),
      jnp.asarray(max_dual),
  )
  return updated_dual, violation

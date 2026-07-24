"""Dual optimization utilities for fixed-KL-radius domain randomization."""

import jax.numpy as jnp


def exponential_moving_average(previous_value, current_value, decay):
  """Updates an exponential moving average."""
  decay = jnp.asarray(decay)
  return (
      decay * jnp.asarray(previous_value)
      + (1.0 - decay) * jnp.asarray(current_value)
  )


def clipped_kl_violation(
    estimated_kl,
    kl_radius,
    max_abs_violation=None,
):
  """Returns KL - radius, optionally clipped to a symmetric interval."""
  violation = jnp.asarray(estimated_kl) - jnp.asarray(kl_radius)
  if max_abs_violation is not None:
    max_abs_violation = jnp.asarray(max_abs_violation)
    violation = jnp.clip(
        violation, -max_abs_violation, max_abs_violation
    )
  return violation


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
    max_abs_violation=None,
):
  """Performs projected ascent using a possibly clipped KL violation."""
  violation = clipped_kl_violation(
      estimated_kl, kl_radius, max_abs_violation
  )
  updated_dual = jnp.clip(
      jnp.asarray(dual_lambda) + jnp.asarray(learning_rate) * violation,
      jnp.asarray(min_dual),
      jnp.asarray(max_dual),
  )
  return updated_dual, violation


def projected_direct_beta_update(
    beta,
    estimated_kl,
    kl_radius,
    learning_rate,
    min_beta,
    max_beta,
    max_abs_violation=None,
):
  """Performs a naive projected beta update using the KL violation.

  This deliberately treats beta as the directly controlled scalar and does not
  apply the chain-rule factor 1 / beta**2.
  """
  violation = clipped_kl_violation(
      estimated_kl, kl_radius, max_abs_violation
  )
  updated_beta = jnp.clip(
      jnp.asarray(beta) + jnp.asarray(learning_rate) * violation,
      jnp.asarray(min_beta),
      jnp.asarray(max_beta),
  )
  return updated_beta, violation

# Modified for the RSDR distribution. See THIRD_PARTY_NOTICES.md.

# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training gradient utility functions."""

from typing import Callable, Optional, Sequence, Union

import jax
import optax
import jax.numpy as jnp

def loss_and_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  g = jax.value_and_grad(loss_fn, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h

def loss_and_forward_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  @jax.custom_jvp
  def f_wrapped(p):
      return loss_fn(p)

  @f_wrapped.defjvp
  def f_wrapped_jvp(primals, tangents):
      (p,), (tp,) = primals, tangents
      # Forward-mode through env (works with dynamic loops)
      val, dval = jax.jvp(loss_fn, (p,), (tp,))
      return val, dval
  @jax.jit
  def value_and_grad(p):
      return jax.value_and_grad(f_wrapped, has_aux=has_aux)(p)
  g = value_and_grad

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h
def multi_loss_and_pgrad(
    loss_fn,
    pmap_axis_name,
    argnums,
    has_aux=False,
):
  g = jax.value_and_grad(loss_fn, argnums=argnums, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h
def global_l2_norm(tree):
    # Works for any PyTree of arrays (params, grads, …)
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  """Wrapper of the loss function that apply gradient updates.

  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    has_aux: Whether the loss_fn has auxiliary data.

  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """
  loss_and_pgrad_fn = loss_and_pgrad(
      loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
  )

  def f(*args, optimizer_state):
    value, grads = loss_and_pgrad_fn(*args)
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, global_l2_norm(grads), params, optimizer_state

  return f

PyTree = any

def _global_l2_norm_multi(grad_list: Sequence[PyTree]) -> jnp.ndarray:
  # leaves = []
  # for g in grad_list:
  #   leaves.extend(jax.tree_util.tree_leaves(g))
  # print("grad list", len(grad_list))
  return [jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(g))) for g in grad_list]

def multi_gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizers: Union[optax.GradientTransformation, Sequence[optax.GradientTransformation]],
    pmap_axis_name: Optional[str],
    argnums: Union[int, Sequence[int]],
    has_aux: bool = False,
):
  """Create an update fn for loss_fn with grads wrt multiple param args.

  Args:
    loss_fn: Callable(params_0, ..., *others) -> loss (and optional aux).
    optimizers: Single optax optimizer or a sequence aligned with `argnums`.
    pmap_axis_name: If not None, grads are reduced via lax.pmean over this axis.
    argnums: Single int or sequence of ints indicating which *positional* args
      are treated as parameters to differentiate and update.
    has_aux: Whether loss_fn returns (loss, aux).

  Returns:
    f(*args, optimizer_states) -> (value_or_(value,aux), global_grad_norm, new_params_list, new_optimizer_states)
      - `optimizer_states` must be a single state (if single optimizer given and single argnum)
        or a sequence aligned with `argnums` (and `optimizers` if a sequence was given).
      - `new_params_list` is a tuple of updated param PyTrees aligned with `argnums`.
  """
  # Normalize argnums to a tuple
  if isinstance(argnums, int):
    argnums = (argnums,)
  else:
    argnums = tuple(argnums)

  # Normalize optimizers to a list aligned with argnums
  if isinstance(optimizers, optax.GradientTransformation):
    optimizers = [optimizers] * len(argnums)
  else:
    assert len(optimizers) == len(argnums), "`optimizers` must align with `argnums`"

  # Build value+grad function across multiple argnums
  g = jax.value_and_grad(loss_fn, argnums=argnums, has_aux=has_aux)

  def loss_and_pgrad_multi(*args, **kwargs):
    value, grads = g(*args, **kwargs)  # grads is a PyTree or tuple of PyTrees aligned with argnums
    if pmap_axis_name is not None:
      # pmean each grad PyTree across devices
      if len(argnums) == 1:
        grads = jax.lax.pmean(grads, axis_name=pmap_axis_name)
      else:
        grads = tuple(jax.lax.pmean(g_i, axis_name=pmap_axis_name) for g_i in grads)
    return value, grads

  def f(*args, optimizer_states):
    # Normalize optimizer_states to a tuple aligned with argnums
    if len(argnums) == 1 and not isinstance(optimizer_states, (tuple, list)):
      optimizer_states = (optimizer_states,)
    else:
      optimizer_states = tuple(optimizer_states)
      assert len(optimizer_states) == len(argnums), "`optimizer_states` must align with `argnums`"

    value, grads = loss_and_pgrad_multi(*args)

    # Ensure grads is a tuple aligned with argnums
    if len(argnums) == 1:
      grads = (grads,)

    # Compute global grad L2 norm across all param sets
    global_norm = _global_l2_norm_multi(grads)

    # Apply updates per (param, grad, optimizer)
    args_list = list(args)
    new_params_list = []
    new_opt_states = []

    for i, (arg_idx, opt, state, grad) in enumerate(zip(argnums, optimizers, optimizer_states, grads)):
      updates, new_state = opt.update(grad, state)
      new_params = optax.apply_updates(args_list[arg_idx], updates)
      args_list[arg_idx] = new_params
      new_params_list.append(new_params)
      new_opt_states.append(new_state)

    # Return same value shape as loss_fn: loss or (loss, aux)
    return value, global_norm, tuple(new_params_list), tuple(new_opt_states)

  return f
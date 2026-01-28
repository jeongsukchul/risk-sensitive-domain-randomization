import contextlib
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from mujoco import mjx
from custom_envs import mjx_env
from brax.envs.base import Wrapper, Env, State
from brax.base import System
import numpy as np
import functools

class TransitionwithParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: Dict[str, Any] = {}

def wrap_for_adv_training(
    env: mjx_env.MjxEnv,
    param_size: int,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    dr_range_low: jnp.ndarray = None,
    dr_range_high: jnp.ndarray = None,
    full_reset: bool = False,
    get_grad = False,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate,
      should equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  env = AdVmapWrapper(env, randomization_fn, param_size, dr_range_low, dr_range_high, get_grad)
  env = EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env, full_reset=full_reset)
  return env
class AdVmapWrapper(Wrapper):
  """Wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[System], Tuple[System, System]],
      param_size: int,
      dr_range_low: jnp.ndarray = None,
      dr_range_high: jnp.ndarray = None,
      get_grad = False,
  ):
    super().__init__(env)
    self.rand_fn = functools.partial(randomization_fn, model=self.mjx_model, rng=None)
    self.get_grad =  get_grad
    self.param_size = param_size
    self.dr_range_low = dr_range_low
    self.dr_range_high = dr_range_high

  @contextlib.contextmanager
  def v_env_fn(self, mjx_model: mjx.Model):
    env = self.env.unwrapped
    old_mjx_model = env._mjx_model
    try:
      env.unwrapped._mjx_model = mjx_model
      yield env
    finally:
      env.unwrapped._mjx_model = old_mjx_model
      
  def reset(self, rng: jax.Array) -> mjx_env.State:
    # state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    def dr_reset(rng):
      param_rng, rng = jax.random.split(rng)
      params = jax.random.uniform(param_rng, (self.param_size,), minval=self.dr_range_low, maxval=self.dr_range_high)
      mjx_model, inaxes = self.rand_fn(params=params)
      with self.v_env_fn(mjx_model) as v_env:
        return v_env.reset(rng), params
    state, params = jax.vmap(dr_reset, )(rng)
    state.info['dr_params'] = params

    if self.get_grad:
      state.info['grad']=jax.tree_util.tree_map(lambda x: jnp.zeros((x.shape + (self.param_size,))), state.obs)
    return state

  def step(self, state: mjx_env.State, action: jax.Array, params: jax.Array) -> State:
    def step(params, s, a):
      if params is None:
        return self.env.step(s,a)
      mjx_model, inaxes = self.rand_fn(params=params)
      with self.v_env_fn(mjx_model) as v_env:
        return v_env.step(s, a)
    ns = jax.vmap(step)(
      params, state, action
    )
    ns.info['dr_params'] = params

    return ns
    
class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jnp.zeros(rng.shape[:-1])
    state.info['truncation'] = jnp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array, params: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action, params)
      return nstate, nstate.reward

    
    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    rewards = jnp.sum(rewards,axis=0)
    state = state.replace(reward=rewards)
    steps = state.info['steps'] + self.action_repeat
    one = jnp.ones_like(state.done)
    zero = jnp.zeros_like(state.done)
    episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
    done = jnp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jnp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps
    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_reward'] += rewards
    state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
    state.info['episode_metrics']['length'] += self.action_repeat
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if metric_name != 'reward':
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
    state.info['episode_done'] = done
    return state.replace(done=done)

class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""
  def __init__(self, env: Any, full_reset: bool = False):
    super().__init__(env)
    self._full_reset = full_reset
    self._info_key = 'AutoResetWrapper'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng_key = jax.vmap(jax.random.split)(rng)
    rng, key = rng_key[..., 0], rng_key[..., 1]
    state = self.env.reset(key)
    state.info[f'{self._info_key}_first_data'] = state.data
    state.info[f'{self._info_key}_first_obs'] = state.obs
    state.info[f'{self._info_key}_rng'] = rng
    state.info[f'{self._info_key}_done_count'] = jnp.zeros(
        key.shape[:-1], dtype=int
    )
    return state
  def step(self, state: mjx_env.State, action: jax.Array, params : jax.Array) -> mjx_env.State:
    # grab the reset state.
    reset_state = None
    rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
    reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
    if self._full_reset:
      reset_state = self.reset(reset_key)
      reset_data = reset_state.data
      reset_obs = reset_state.obs
    else:
      reset_data = state.info[f'{self._info_key}_first_data']
      reset_obs = state.info[f'{self._info_key}_first_obs']

    if 'steps' in state.info:
      # reset steps to 0 if done.
      steps = state.info['steps']
      steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
      state.info.update(steps=steps)

    state = state.replace(done=jnp.zeros_like(state.done))
    state = self.env.step(state, action, params)
    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jnp.where(done, x, y)

    data = jax.tree.map(where_done, reset_data, state.data)
    obs = jax.tree.map(where_done, reset_obs, state.obs)

    next_info = state.info
    done_count_key = f'{self._info_key}_done_count'
    if self._full_reset and reset_state:
      next_info = jax.tree.map(where_done, reset_state.info, state.info)
      next_info[done_count_key] = state.info[done_count_key]

      if 'steps' in next_info:
        next_info['steps'] = state.info['steps']
      preserve_info_key = f'{self._info_key}_preserve_info'
      if preserve_info_key in next_info:
        next_info[preserve_info_key] = state.info[preserve_info_key]

    next_info[done_count_key] += state.done.astype(int)
    next_info[f'{self._info_key}_rng'] = reset_rng

    return state.replace(data=data, obs=obs, info=next_info)

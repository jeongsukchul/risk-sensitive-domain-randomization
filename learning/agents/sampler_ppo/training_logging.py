"""Helpers for selecting buffered training metrics to log."""

from typing import Tuple


def training_log_offsets(
    start_update_step: int,
    num_steps: int,
    training_log_freq: int,
    exclude_final_step: bool,
) -> Tuple[int, ...]:
  """Returns zero-based metric offsets aligned to global training updates."""
  if training_log_freq <= 0:
    return ()
  return tuple(
      offset
      for offset in range(num_steps)
      if (start_update_step + offset + 1) % training_log_freq == 0
      and not (exclude_final_step and offset == num_steps - 1)
  )

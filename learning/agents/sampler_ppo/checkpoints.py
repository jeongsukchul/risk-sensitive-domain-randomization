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


"""Serialize sampler state without duplicating method dispatch in the trainer."""

from pathlib import Path

import flax
import jax
from absl import logging

_SAMPLER_STATE_FIELDS = {
    "AutoDR": "autodr_state",
    "DORAEMON": "doraemon_state",
    "GBS": "flow_state",
    "GMM": "gmm_training_state",
}


def save_sampler_checkpoint(path, sampler_choice, training_state):
    """Write the first replica's sampler state, keeping existing msgpack layouts.

    Uniform/nominal PPO and EPOpt have no sampler state to save. Flow sampler
    checkpointing remains unsupported, as in the original trainer.
    """
    field = _SAMPLER_STATE_FIELDS.get(sampler_choice)
    if field is None:
        return
    state = jax.tree_util.tree_map(lambda leaf: leaf[0], getattr(training_state, field))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(state))
    logging.info("Saved %s state to %s", sampler_choice, path)

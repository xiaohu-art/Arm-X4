# Copyright (c) 2026, PBC, Tsinghua University.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules import ActorCritic

class ActorCritic(ActorCritic):
    """Customized Actor-critic Module."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__(
            obs, 
            obs_groups, 
            num_actions, 
            actor_obs_normalization, 
            critic_obs_normalization, 
            actor_hidden_dims, 
            critic_hidden_dims, 
            activation, 
            init_noise_std, 
            noise_std_type, 
            state_dependent_std
        )
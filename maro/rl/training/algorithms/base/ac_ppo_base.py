# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl.model import VNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.rollout import ExpElement
from maro.rl.training import AbsTrainOps, FIFOReplayMemory, RemoteOps, SingleAgentTrainer, TrainerParams, remote
from maro.rl.utils import (
    TransitionBatch, discount_cumsum, get_torch_device, merge_transition_batches, ndarray_to_tensor
)


@dataclass
class DiscreteACBasedParams(TrainerParams, metaclass=ABCMeta):
    """
    Parameter bundle for discrete actor-critic based algorithms (discrete actor-critic & discrete PPO)

    get_v_critic_net_func (Callable[[], VNet]): Function to get V critic net.
    reward_discount (float, default=0.9): Reward decay as defined in standard RL terminology.
    grad_iters (int, default=1): Number of iterations to calculate gradients.
    critic_loss_cls (Callable, default=None): Critic loss function. If it is None, use MSE.
    lam (float, default=0.9): Lambda value for generalized advantage estimation (TD-Lambda).
    min_logp (float, default=None): Lower bound for clamping logP values during learning.
        This is to prevent logP from becoming very large in magnitude and causing stability issues.
        If it is None, it means no lower bound.
    """
    get_v_critic_net_func: Callable[[], VNet] = None
    reward_discount: float = 0.9
    grad_iters: int = 1
    critic_loss_cls: Callable = None
    lam: float = 0.9
    min_logp: Optional[float] = None


class DiscreteACBasedOps(AbsTrainOps):
    """Base class of discrete actor-critic algorithm implementation. Reference: https://tinyurl.com/2ezte4cr
    """
    def __init__(
        self,
        name: str,
        policy_creator: Callable[[str], DiscretePolicyGradient],
        get_v_critic_net_func: Callable[[], VNet],
        parallelism: int = 1,
        *,
        reward_discount: float = 0.9,
        critic_loss_cls: Callable = None,
        clip_ratio: float = 0.1,
        lam: float = 0.9,
        min_logp: float = None,
    ) -> None:
        super(DiscreteACBasedOps, self).__init__(
            name=name,
            policy_creator=policy_creator,
            parallelism=parallelism,
        )

        assert isinstance(self._policy, DiscretePolicyGradient)

        self._reward_discount = reward_discount
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._min_logp = min_logp
        self._v_critic_net = get_v_critic_net_func()

        self._device = None

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the critic loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
        self._v_critic_net.train()
        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        state_values = self._v_critic_net.v_values(states)
        returns = ndarray_to_tensor(batch.returns, device=self._device)
        return self._critic_loss_func(state_values, returns)

    @remote
    def get_critic_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        """Compute the critic network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The critic gradient of the batch.
        """
        return self._v_critic_net.get_gradients(self._get_critic_loss(batch))

    def update_critic(self, batch: TransitionBatch) -> None:
        """Update the critic network using a batch.

        Args:
            batch (TransitionBatch): Batch.
        """
        self._v_critic_net.step(self._get_critic_loss(batch))

    def update_critic_with_grad(self, grad_dict: dict) -> None:
        """Update the critic network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._v_critic_net.train()
        self._v_critic_net.apply_gradients(grad_dict)

    def _get_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the actor loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The actor loss of the batch.
        """
        assert isinstance(self._policy, DiscretePolicyGradient)

        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        actions = ndarray_to_tensor(batch.actions, device=self._device).long()  # a
        advantages = ndarray_to_tensor(batch.advantages, device=self._device)

        self._policy.train()
        action_probs = self._policy.get_action_probs(states)
        dist = Categorical(action_probs)
        dist_entropy = dist.entropy()
        logps = torch.log(action_probs.gather(1, actions).squeeze())
        logps = torch.clamp(logps, min=self._min_logp, max=.0)
        if self._clip_ratio is not None:
            logps_old = ndarray_to_tensor(batch.old_logps, device=self._device)
            ratio = torch.exp(logps - logps_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages))
        else:
            actor_loss = -(logps * advantages)  # I * delta * log pi(a|s)
        loss = (actor_loss - 0.2*dist_entropy).mean()
        print('actor_loss: ', actor_loss.mean(), 'entropy loss: ', dist_entropy.mean())
        return loss

    @remote
    def get_actor_grad(self, batch: TransitionBatch) -> Dict[str, torch.Tensor]:
        """Compute the actor network's gradients of a batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            grad (torch.Tensor): The actor gradient of the batch.
        """
        return self._policy.get_gradients(self._get_actor_loss(batch))

    def update_actor(self, batch: TransitionBatch) -> None:
        """Update the actor network using a batch.

        Args:
            batch (TransitionBatch): Batch.
        """
        self._policy.train_step(self._get_actor_loss(batch))

    def update_actor_with_grad(self, grad_dict: dict) -> None:
        """Update the actor network with remotely computed gradients.

        Args:
            grad_dict (dict): Gradients.
        """
        self._policy.train()
        self._policy.apply_gradients(grad_dict)

    def get_state(self) -> dict:
        return {
            "policy": self._policy.get_state(),
            "critic": self._v_critic_net.get_state(),
        }

    def set_state(self, ops_state_dict: dict) -> None:
        self._policy.set_state(ops_state_dict["policy"])
        self._v_critic_net.set_state(ops_state_dict["critic"])

    def _preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """Preprocess the batch to get the returns & advantages.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            The updated batch.
        """
        assert isinstance(batch, TransitionBatch)
        # Preprocess returns
        batch.calc_returns(self._reward_discount)

        # Preprocess advantages
        states = ndarray_to_tensor(batch.states, device=self._device)  # s
        actions = ndarray_to_tensor(batch.actions.astype(np.int64), device=self._device)  # a

        values = self._v_critic_net.v_values(states).detach().cpu().numpy()
        values = np.concatenate([values, values[-1:]])
        rewards = np.concatenate([batch.rewards, values[-1:]])
        deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]  # r + gamma * v(s') - v(s)
        advantages = discount_cumsum(deltas, self._reward_discount * self._lam)
        batch.advantages = advantages

        if self._clip_ratio is not None:
            batch.old_logps = self._policy.get_state_action_logps(states, actions).detach().numpy()

        return batch

    def preprocess_and_merge_batches(self, batch_list: List[TransitionBatch]) -> TransitionBatch:
        """Preprocess and merge a list of transition batches to a single transition batch.

        Args:
            batch_list (List[TransitionBatch]): List of batches.

        Returns:
            The merged batch.
        """
        return merge_transition_batches([self._preprocess_batch(batch) for batch in batch_list])

    def to_device(self, device: str = None) -> None:
        self._device = get_torch_device(device)
        self._policy.to_device(self._device)
        self._v_critic_net.to(self._device)


class DiscretePPOBasedOps(DiscreteACBasedOps):
    """Base class of discrete actor-critic algorithm implementation. Reference: https://tinyurl.com/2ezte4cr
    """
    def __init__(
        self,
        name: str,
        policy_creator: Callable[[str], DiscretePolicyGradient],
        get_v_critic_net_func: Callable[[], VNet],
        parallelism: int = 1,
        *,
        reward_discount: float = 0.9,
        critic_loss_cls: Callable = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        min_logp: float = None,
    ) -> None:
        super(DiscretePPOBasedOps, self).__init__(
            name=name,
            policy_creator=policy_creator,
            get_v_critic_net_func=get_v_critic_net_func,
            parallelism=parallelism,
            reward_discount=reward_discount,
            critic_loss_cls=critic_loss_cls,
            clip_ratio=clip_ratio,
            lam=lam,
            min_logp=min_logp,
        )

        assert isinstance(self._policy, DiscretePolicyGradient)
        self._policy_old = self._policy_creator(self._name)
        self._policy_old.set_state(self._policy.get_state())

    def _preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """Preprocess the batch to get the returns & advantages.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            The updated batch.
        """
        assert isinstance(batch, TransitionBatch)
        # Preprocess returns
        batch.calc_returns(self._reward_discount)

        # Preprocess advantages
        states = ndarray_to_tensor(batch.states, self._device)  # s
        state_values = self._v_critic_net.v_values(states).detach().numpy()
        values = np.concatenate([state_values[1:], np.zeros(1).astype(np.float32)])
        batch.advantages = (batch.rewards+self._reward_discount*values - state_values)
        return batch

    def _get_critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the critic loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The critic loss of the batch.
        """
        self._v_critic_net.train()
        states = ndarray_to_tensor(batch.states, self._device)  # s
        state_values = self._v_critic_net.v_values(states)
        values = state_values.detach().numpy()
        values = np.concatenate([values[1:], np.zeros(1).astype(np.float32)])
        returns = batch.rewards + self._reward_discount * values
        returns = ndarray_to_tensor(returns, self._device)
        return self._critic_loss_func(state_values, returns)

    def _get_actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Compute the actor loss of the batch.

        Args:
            batch (TransitionBatch): Batch.

        Returns:
            loss (torch.Tensor): The actor loss of the batch.
        """
        assert isinstance(self._policy, DiscretePolicyGradient)

        states = ndarray_to_tensor(batch.states, self._device)  # s
        actions = ndarray_to_tensor(batch.actions, self._device).long()  # a
        advantages = ndarray_to_tensor(batch.advantages, self._device)

        if self._clip_ratio is not None:
            self._policy_old.eval()
            logps_old = self._policy_old.get_state_action_logps(states, actions).detach()
        else:
            logps_old = None

        self._policy.train()
        action_probs = self._policy.get_action_probs(states)
        dist = Categorical(action_probs)
        # print('probs: ', action_probs)
        dist_entropy = dist.entropy()
        logps = torch.log(action_probs.gather(1, actions).squeeze())
        logps = torch.clamp(logps, min=self._min_logp, max=.0)
        if self._clip_ratio is not None:
            ratio = torch.exp(logps - logps_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages))
        else:
            actor_loss = -(logps * advantages)  # I * delta * log pi(a|s)
        loss = (actor_loss - 0.2*dist_entropy).mean()
        print('actor_loss: ', actor_loss.mean(), 'entropy loss: ', dist_entropy.mean())
        return loss


class DiscreteACBasedTrainer(SingleAgentTrainer):
    """Base class of discrete actor-critic algorithm implementation.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    """

    def __init__(self, name: str, params: DiscreteACBasedParams) -> None:
        super(DiscreteACBasedTrainer, self).__init__(name, params)
        self._params = params
        self._replay_memory_dict: Dict[Any, FIFOReplayMemory] = {}

    def build(self) -> None:
        self._ops = self.get_ops()
        self._replay_memory_dict = collections.defaultdict(lambda: FIFOReplayMemory(
            capacity=self._params.replay_memory_capacity,
            state_dim=self._ops.policy_state_dim,
            action_dim=self._ops.policy_action_dim,
        ))

    def record(self, env_idx: int, exp_element: ExpElement) -> None:
        for agent_name in exp_element.agent_names:
            memory = self._replay_memory_dict[(env_idx, agent_name)]
            transition_batch = TransitionBatch(
                states=np.expand_dims(exp_element.agent_state_dict[agent_name], axis=0),
                actions=np.expand_dims(exp_element.action_dict[agent_name], axis=0),
                rewards=np.array([exp_element.reward_dict[agent_name]]),
                terminals=np.array([exp_element.terminal_dict[agent_name]]),
                next_states=np.expand_dims(
                    exp_element.next_agent_state_dict.get(agent_name, exp_element.agent_state_dict[agent_name]),
                    axis=0,
                ),
            )
            memory.put(transition_batch)

    def get_local_ops(self) -> AbsTrainOps:
        return DiscreteACBasedOps(
            name=self._policy_name,
            policy_creator=self._policy_creator,
            parallelism=self._params.data_parallelism,
            **self._params.extract_ops_params(),
        )

    def _get_batch(self) -> TransitionBatch:
        batch_list = [memory.sample(-1) for memory in self._replay_memory_dict.values()]
        return self._ops.preprocess_and_merge_batches(batch_list)

    def train_step(self) -> None:
        assert isinstance(self._ops, DiscreteACBasedOps)
        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)

    async def train_step_as_task(self) -> None:
        assert isinstance(self._ops, RemoteOps)
        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic_with_grad(await self._ops.get_critic_grad(batch))
            self._ops.update_actor_with_grad(await self._ops.get_actor_grad(batch))

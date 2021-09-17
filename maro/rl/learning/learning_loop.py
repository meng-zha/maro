# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import getcwd
from typing import Callable, List, Union

from maro.utils import Logger

from .env_sampler import AbsEnvSampler
from .helpers import get_eval_schedule, get_rollout_finish_msg
from .policy_manager import AbsPolicyManager
from .rollout_manager import AbsRolloutManager


def learn(
    get_rollout_manager: Callable[[], Union[AbsEnvSampler, AbsRolloutManager]],
    num_episodes: int,
    num_steps: int = -1,
    get_policy_manager: Callable[[], AbsPolicyManager] = None,
    eval_schedule: Union[int, List[int]] = None,
    eval_after_last_episode: bool = False,
    post_collect: Callable = None,
    post_evaluate: Callable = None,
    log_dir: str = getcwd()
):
    """Run the main learning loop in single-threaded or distributed mode.

    In distributed mode, this is the main process that executes 2-phase learning cycles: simulation data collection
    and policy update. The transition from one phase to another is synchronous.

    Args:
        get_rollout_manager (AbsRolloutManager): Function to create an ``AbsEnvSampler`` or ``AbsRolloutManager``
            instance to control the data collecting phase of the learning cycle. The function takes no parameters
            and returns an ``AbsRolloutManager``. Use this for multi-process or distributed policy training.
        num_episodes (int): Number of learning episodes. The environment always runs to completion in each episode.
        num_steps (int): Number of environment steps to roll out each time. Defaults to -1, in which
            case the roll-out will be executed until the end of the environment.
        get_policy_manager (AbsPolicyManager): Function to create an ``AbsPolicyManager`` instance to control policy
            update phase of the learning cycle. The function takes no parameters and returns an ``AbsPolicyManager``.
            Use this for multi-process or distributed data collection. Defaults to None.
        eval_schedule (Union[int, List[int]]): Evaluation schedule. If an integer is provided, the policies will
            will be evaluated every ``eval_schedule`` episodes. If a list is provided, the policies will be evaluated
            at the end of the episodes given in the list. Defaults to None, in which case no evaluation is performed
            unless ``eval_after_last_episode`` is set to True.
        eval_after_last_episode (bool): If True, the policy will be evaluated after the last episode of learning is
            finished. Defaults to False.
        early_stopper (AbsEarlyStopper): Early stopper to stop the main training loop if certain conditions on the
            environment metric are met following an evaluation episode. Default to None.
        post_collect (Callable): Custom function to process whatever information is collected by each
            environment wrapper (local or remote) at the end of ``collect`` calls. The function signature should
            be (trackers, ep, segment) -> None, where tracker is a list of environment wrappers' ``tracker`` members.
            Defaults to None.
        post_evaluate (Callable): Custom function to process whatever information is collected by each
            environment wrapper (local or remote) at the end of ``evaluate`` calls. The function signature should
            be (trackers, ep) -> None, where tracker is a list of environment wrappers' ``tracker`` members. Defaults
            to None.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "LEARNER" will be created at init time
            and this directory will be used to save the log files generated by it. Defaults to the current working
            directory.
    """
    if num_steps == 0 or num_steps < -1:
        raise ValueError("num_steps must be a positive integer or -1")

    rollout_manager = get_rollout_manager()
    if not get_policy_manager:
        assert isinstance(rollout_manager, AbsEnvSampler), \
            "'get_rollout_manager' must return an 'AbsEnvSampler' if 'get_policy_manager' is None."

    policy_manager = get_policy_manager() if get_policy_manager else None
    logger = Logger("LEARNER", dump_folder=log_dir)
    # evaluation schedule
    eval_schedule = get_eval_schedule(eval_schedule, num_episodes, final=eval_after_last_episode)
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0

    def collect_and_update():
        collect_time = policy_update_time = 0
        if isinstance(rollout_manager, AbsRolloutManager):
            rollout_manager.reset()
        segment, end_of_episode = 1, False
        while not end_of_episode:
            # experience collection
            tc0 = time.time()
            if policy_manager:
                policy_state_dict = policy_manager.get_state()
                rollout_info_by_policy, trackers = rollout_manager.collect(ep, segment, policy_state_dict)
                end_of_episode = rollout_manager.end_of_episode
            else:
                result = rollout_manager.sample(num_steps=num_steps, return_rollout_info=False)
                trackers = [result["tracker"]]
                logger.info(
                    get_rollout_finish_msg(ep, result["step_range"], exploration_params=result["exploration_params"])
                )
                end_of_episode = result["end_of_episode"]

            if post_collect:
                post_collect(trackers, ep, segment)

            collect_time += time.time() - tc0
            tu0 = time.time()
            if policy_manager:
                policy_manager.update(rollout_info_by_policy)
            else:
                rollout_manager.agent_wrapper.improve()
            policy_update_time += time.time() - tu0
            segment += 1

        # performance details
        logger.info(f"ep {ep} summary - collect time: {collect_time}, policy update time: {policy_update_time}")

    for ep in range(1, num_episodes + 1):
        collect_and_update()
        if ep == eval_schedule[eval_point_index]:
            eval_point_index += 1
            if isinstance(rollout_manager, AbsEnvSampler):
                trackers = [rollout_manager.test()]
            else:
                trackers = rollout_manager.evaluate(ep, policy_manager.get_state())
            if post_evaluate:
                post_evaluate(trackers, ep)

    if isinstance(rollout_manager, AbsRolloutManager):
        rollout_manager.exit()

    if hasattr(policy_manager, "exit"):
        policy_manager.exit()

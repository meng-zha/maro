# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import dirname, join, realpath

from maro.rl.workflows.scenario import Scenario

# config variables
SCENARIO_NAME = "cim"
SCENARIO_PATH = join(dirname(dirname(realpath(__file__))), SCENARIO_NAME, "rl")

if __name__ == "__main__":
    scenario = Scenario(SCENARIO_PATH)
    policy_creator = scenario.policy_creator
    policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
    policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}

    env_sampler = scenario.env_sampler_creator(policy_creator)
    result = env_sampler.eval()
    if scenario.post_evaluate:
        scenario.post_evaluate(result["info"], 0)
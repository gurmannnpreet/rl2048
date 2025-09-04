import numpy as np

from rl2048.envs.game2048 import Game2048Env


def test_env_basic_step():
    env = Game2048Env(size=4, target=32)
    obs, info = env.reset(seed=123)
    assert obs.shape == (4, 4)
    assert (obs == 0).sum() >= 14  # two tiles spawned
    obs2, reward, terminated, truncated, info2 = env.step(2)  # left
    assert obs2.shape == (4, 4)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "valid_actions" in info2


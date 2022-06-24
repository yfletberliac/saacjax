# NOTE: this code was mainly taken from:
# https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
import gym
import numpy as np
from dm_control import suite
from dm_env import specs
from gym import core, spaces

from rljax.env.atari import FrameStack
import realworldrl_suite.environments as rwrl
import dm2gym.envs.dm_suite_env as dm2gym

gym.logger.set_level(40)


def make_dmc_env(domain_name, task_name, combined_challenge, seed):

    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        combined_challenge=combined_challenge,
        environment_kwargs=dict(log_every=20, log_safety_vars=True, flat_observation=True),
        random=seed,
        safety_spec=dict(enable=True, observations=True, safety_coeff=0.7),
    )

    env = GymEnv(env)
    print("Obs space: ", env.observation_space)
    print("Action space: ", env.action_space)

    if not hasattr(env, "_max_episode_steps"):
        setattr(env, "_max_episode_steps", 1000)

    return env


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


def make(
    domain_name,
    task_name,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
):
    env_id = "dmc_%s_%s_%s_%s_%s-v1" % (domain_name, task_name, seed, height, width)

    if from_pixels:
        assert not visualize_reward, "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        gym.envs.registration.register(
            id=env_id,
            entry_point=DMCWrapper,
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)


class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
    ):
        assert "random" in task_kwargs, "please specify a seed, for deterministic behaviour"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip

        # Create task.
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )

        # True action space.
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._low = self._true_action_space.low
        self._delta = self._true_action_space.high - self._true_action_space.low
        # Normalized action space.
        self._norm_action_space = spaces.Box(low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float64)

        # Create observation space.
        if from_pixels:
            self._observation_space = spaces.Box(low=0, high=255, shape=[height, width, 3], dtype=np.uint8)
        else:
            self._observation_space = _spec_to_box(self._env.observation_spec().values())

        self._state_space = _spec_to_box(self._env.observation_spec().values())
        self.current_state = None

        # Set seed.
        self.seed(seed=task_kwargs.get("random", 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(height=self._height, width=self._width, camera_id=self._camera_id)
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        return (action + 1.0) * self._delta / 2.0 + self._low

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        action = np.clip(
            action,
            a_min=self._norm_action_space.low,
            a_max=self._norm_action_space.high,
        )
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0.0

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        return obs, reward, done, {}

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        return self._get_obs(time_step)

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)


class GymEnv(dm2gym.DMSuiteEnv):
  """Wrapper that convert a realworldrl environment to a gym environment."""

  def __init__(self, env):
    """Constructor. We reuse the facilities from dm2gym."""
    self.env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': round(1. / self.env.control_timestep())
    }
    self.observation_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.observation_spec())
    self.action_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.action_spec())
    self.viewer = None

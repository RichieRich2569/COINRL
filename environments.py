import numpy as np
import math
import gymnasium as gym
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium.error import DependencyNotInstalled
import os.path as path

# TimeLimitMixin – adds internal step-counting & truncation
class TimeLimitMixin:
    """
    Adds Gymnasium-style truncation without relying on the TimeLimit wrapper.
    `self._elapsed_steps` is reset to 0 on `reset()` and incremented on every
    `step()`.  `truncated` is True when the counter reaches
    `self._max_episode_steps`.
    """
    def __init__(self, *args, max_episode_steps: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)          # ──> call parent __init__
        # Prefer a user-supplied limit, then the env spec, else a sane default
        self._max_episode_steps = (
            max_episode_steps
            or (self.spec.max_episode_steps if self.spec is not None else 500)
        )
        self._elapsed_steps = 0

    # -------------------------------------------------
    # standard Gymnasium API
    # -------------------------------------------------
    def reset(self, *args, **kwargs):
        self._elapsed_steps = 0
        return super().reset(*args, **kwargs)

    def step(self, action):
        # Call the parent env to get obs, reward, terminated, truncated, info
        obs, reward, terminated, _, info = super().step(action)

        # bookkeeping
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self._max_episode_steps

        return obs, reward, terminated, truncated, info


class CustomCartPoleEnv(TimeLimitMixin, CartPoleEnv):
    """CartPole with tunable physics."""

    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,       # actually half the pole length
        force_mag: float = 10.0,
        tau: float = 0.02,
        theta_threshold_radians: float | None = None,
        x_threshold: float | None = None,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        TimeLimitMixin.__init__(self, max_episode_steps=max_episode_steps)
        CartPoleEnv.__init__(self, render_mode=kwargs.pop("render_mode", None))

        # --- user‑defined parameters ---
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.length = length
        self.force_mag = force_mag
        self.tau = tau
        if theta_threshold_radians is not None:
            self.theta_threshold_radians = theta_threshold_radians
        if x_threshold is not None:
            self.x_threshold = x_threshold

        # --- recomputed derived values ---
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length


class CustomPendulumEnv(TimeLimitMixin, PendulumEnv):
    """Pendulum with adjustable parameters"""

    def __init__(
        self,
        g: float = 9.8,
        m: float = 1.0,
        l: float = 1.0,
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        dt: float = 0.05,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        TimeLimitMixin.__init__(self, max_episode_steps=max_episode_steps)
        PendulumEnv.__init__(self, render_mode=kwargs.pop("render_mode", None))
        self.g = g
        self.m = m
        self.l = l
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt  # NB: PendulumEnv uses _dt internally; update both
        self._dt = dt

        # No extra derived constants required—PendulumEnv queries these directly
        
    def render(self):
        """Render the pendulum using the *current* length self.l."""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You called render without specifying render_mode; "
                'pass render_mode="human" or "rgb_array" when you create the env.'
            )
            return

        # ---- lazy import / surface initialisation --------------------
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            w = int(self.screen_dim)
            h = int(self.screen_dim)
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((w, h))
            else:  # "rgb_array"
                self.screen = pygame.Surface((w, h))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (int(self.screen_dim), int(self.screen_dim))
        )
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = float(self.screen_dim) / (bound * 2)
        offset = int(self.screen_dim // 2)

        rod_length = float(self.l) * scale
        rod_width = 0.2 * scale

        # Four corners of the rod (before rotation)
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]

        # Rotate to current angle θ and translate to screen centre
        theta = self.state[0]
        transformed = [
            (pygame.math.Vector2(c).rotate_rad(theta + np.pi / 2) + (offset, offset))
            for c in coords
        ]
        gfxdraw.aapolygon(self.surf, transformed, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed, (204, 77, 77))

        # Axle
        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        # Bob
        rod_end = pygame.math.Vector2(rod_length, 0).rotate_rad(theta + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, *rod_end, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, *rod_end, int(rod_width / 2), (204, 77, 77))

        img = None
        try:
            # Locate the file inside Gymnasium’s classic-control package
            #   …/gymnasium/envs/classic_control/assets/clockwise.png
            import importlib.resources as resources
            asset_path = resources.files(
                "gymnasium.envs.classic_control"
            ).joinpath("assets/clockwise.png")
            img = pygame.image.load(str(asset_path))
        except (FileNotFoundError, ModuleNotFoundError, pygame.error):
            # No asset?  Omit the arrow.
            img = None

        if img is not None and self.last_u is not None:
            torque = float(np.array(self.last_u).reshape(-1)[0])
            arrow_size = int(scale * abs(torque) / 2)
            arrow_size = max(1, arrow_size)

            arrow = pygame.transform.smoothscale(
                img,
                (int(arrow_size), int(arrow_size)),
            )
            arrow = pygame.transform.flip(arrow, bool(torque > 0), True)
            self.surf.blit(
                arrow,
                (
                    offset - arrow.get_rect().centerx,
                    offset - arrow.get_rect().centery,
                ),
            )

        # Axle cap
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        # Flip Y-axis (pygame’s origin is top-left)
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:                                             # "rgb_array"
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


class CustomAcrobotEnv(TimeLimitMixin, AcrobotEnv):
    """Acrobot with tunable link lengths, masses, and gravity."""

    def __init__(
        self,
        link_length_1: float = 1.0,
        link_length_2: float = 1.0,
        link_mass_1: float = 1.0,
        link_mass_2: float = 1.0,
        link_com_pos_1: float = 0.5,   # COM position wrt link length
        link_com_pos_2: float = 0.5,
        link_moi: float = 1.0,
        max_vel_1: float = 4 * 3.1416,
        max_vel_2: float = 9 * 3.1416,
        torque_mag: float = 1.0,
        gravity: float = 9.8,
        dt: float = 0.2,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        TimeLimitMixin.__init__(self, max_episode_steps=max_episode_steps)
        AcrobotEnv.__init__(self, render_mode=kwargs.pop("render_mode", None))

        # --- user‑defined parameters ---
        self.LINK_LENGTH_1 = link_length_1
        self.LINK_LENGTH_2 = link_length_2
        self.LINK_MASS_1 = link_mass_1
        self.LINK_MASS_2 = link_mass_2
        self.LINK_COM_POS_1 = link_com_pos_1
        self.LINK_COM_POS_2 = link_com_pos_2
        self.LINK_MOI = link_moi
        self.MAX_VEL_1 = max_vel_1
        self.MAX_VEL_2 = max_vel_2
        self.TORQUE_MAG = torque_mag
        self.GRAVITY = gravity
        self.dt = dt

        # --- recomputed helper constants ---
        self.I1 = (
            self.LINK_MASS_1
            * self.LINK_COM_POS_1**2
            + self.LINK_MOI
        )
        self.I2 = (
            self.LINK_MASS_2
            * self.LINK_COM_POS_2**2
            + self.LINK_MOI
        )
        # AcrobotEnv’s dynamics refer directly to these attributes; updating
        # them here suffices.

class CustomMountainCarEnv(TimeLimitMixin, MountainCarEnv):
    """A MountainCar environment that allows customizing the amplitude of the slope."""
    
    def __init__(self, amplitude=1, gravity_sf=1, max_speed_sf=1, force_sf=1, goal_position=0.5, max_episode_steps=None, shaping_coef=0.0, shaping_gamma=0.995, **kwargs):
        """
        amplitude: The 'A' in y = A sin(3x) + offset
        goal_position: The x-position at which the episode terminates successfully.
        goal_velocity: Required velocity upon reaching the goal (default 0).
        shaping_coef: If nonzero, adds potential-based reward shaping
            F = shaping_gamma * Phi(s') - Phi(s) with Phi = shaping_coef * mechanical
            energy (0.5 v^2 + U(x)), the classic aligned dense signal that makes
            MountainCar's sparse reward learnable for policy-gradient methods at short
            episode caps. Potential-based shaping preserves the optimal policy
            (Ng et al., 1999). The unshaped reward is exposed as info["raw_reward"].
        shaping_gamma: Discount used inside the shaping term (match the agent's gamma).
        """
        TimeLimitMixin.__init__(self, max_episode_steps=max_episode_steps)
        MountainCarEnv.__init__(self, render_mode=kwargs.pop("render_mode", None),
                                goal_velocity=kwargs.get("goal_velocity", 0))
        self.amplitude = amplitude
        self.gravity_sf = gravity_sf
        self.max_speed_sf = max_speed_sf
        self.force_sf = force_sf
        self.shaping_coef = float(shaping_coef)
        self.shaping_gamma = float(shaping_gamma)


        self.gravity = amplitude*self.gravity*gravity_sf # Effect of changing slope equivalent to changing gravity
        self.max_speed = self.max_speed*max_speed_sf
        self.force = self.force*force_sf
        self.goal_position = goal_position

    def _potential(self, x, v):
        """Mechanical-energy shaping potential. U(x) = gravity * sin(3x) / 3 is the
        antiderivative of the slope's gravity acceleration term, so Phi tracks the true
        energy the optimal pump strategy accumulates; amplitude 0 leaves only the
        kinetic term."""
        return self.shaping_coef * (0.5 * v * v + self.gravity * np.sin(3.0 * x) / 3.0)

    def step(self, action):
        if self.shaping_coef:
            x0, v0 = float(self.state[0]), float(self.state[1])
            obs, reward, terminated, truncated, info = super().step(action)
            x1, v1 = float(self.state[0]), float(self.state[1])
            info["raw_reward"] = float(reward)
            reward = float(reward) + self.shaping_gamma * self._potential(x1, v1) \
                - self._potential(x0, v0)
            return obs, reward, terminated, truncated, info
        return super().step(action)
    
    def _height(self, xs):
        A = float(self.amplitude)
        return 0.45 * A * np.sin(3 * xs) + 0.1 + 0.45 * abs(A)

    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        if abs(self.amplitude) > 1:
            scale /= abs(self.amplitude)
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.atan(0.45*3*self.amplitude*math.cos(3 * pos)))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.atan(0.45*3*self.amplitude*math.cos(3 * pos)))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

class _PadToAcrobotInterfaceMixin:
    """
    Standardise (obs_dim, n_actions) to match Acrobot:
      - obs padded to 6 (zeros appended)
      - action space promoted to Discrete(3)
      - info["action_mask"] indicates which actions are valid for the underlying env
    """

    TARGET_OBS_DIM = 6
    TARGET_N_ACTIONS = 3

    def _init_acrobot_interface(self):
        # Expect underlying env already set its spaces in __init__
        assert isinstance(self.observation_space, gym.spaces.Box), "Obs space must be Box"
        assert isinstance(self.action_space, gym.spaces.Discrete), "Action space must be Discrete"

        orig_shape = self.observation_space.shape
        assert orig_shape is not None and len(orig_shape) == 1, "Obs must be 1D Box"
        self._orig_obs_dim = int(orig_shape[0])
        self._orig_n_actions = int(self.action_space.n)

        if self._orig_obs_dim > self.TARGET_OBS_DIM:
            raise ValueError(f"orig obs dim {self._orig_obs_dim} > target {self.TARGET_OBS_DIM}")
        if self._orig_n_actions > self.TARGET_N_ACTIONS:
            raise ValueError(f"orig n_actions {self._orig_n_actions} > target {self.TARGET_N_ACTIONS}")

        # Override *exposed* spaces to the shared interface
        low = np.full((self.TARGET_OBS_DIM,), -np.inf, dtype=np.float32)
        high = np.full((self.TARGET_OBS_DIM,),  np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.TARGET_N_ACTIONS)

    def _pad_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        out = np.zeros((self.TARGET_OBS_DIM,), dtype=np.float32)
        out[: self._orig_obs_dim] = obs[: self._orig_obs_dim]
        return out

    def _action_mask(self):
        mask = np.zeros((self.TARGET_N_ACTIONS,), dtype=np.int8)
        mask[: self._orig_n_actions] = 1
        return mask

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info = dict(info) if info is not None else {}
        info["action_mask"] = self._action_mask()
        return self._pad_obs(obs), info

    def step(self, action):
        action = int(action)

        # Remap invalid actions to a valid one (keeps training stable).
        # Alternative: penalise + terminate, if you prefer.
        if action >= self._orig_n_actions:
            action = self._orig_n_actions - 1

        obs, reward, terminated, truncated, info = super().step(action)
        info = dict(info) if info is not None else {}
        info["action_mask"] = self._action_mask()
        return self._pad_obs(obs), reward, terminated, truncated, info


class MountainCarXEnv(_PadToAcrobotInterfaceMixin, CustomMountainCarEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_acrobot_interface()


class CartPoleXEnv(_PadToAcrobotInterfaceMixin, CustomCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_acrobot_interface()


class AcrobotXEnv(_PadToAcrobotInterfaceMixin, CustomAcrobotEnv):
    """
    For consistency: Acrobot already has obs_dim=6, n_actions=3,
    so this is effectively a no-op except for always providing action_mask.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_acrobot_interface()

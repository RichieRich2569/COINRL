import numpy as np
import math
import gymnasium as gym
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium.error import DependencyNotInstalled

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
    
    def __init__(self, amplitude=1, gravity_sf=1, max_speed_sf=1, force_sf=1, goal_position=0.5, max_episode_steps=None, **kwargs):
        """
        amplitude: The 'A' in y = A sin(3x) + offset
        goal_position: The x-position at which the episode terminates successfully.
        goal_velocity: Required velocity upon reaching the goal (default 0).
        """
        TimeLimitMixin.__init__(self, max_episode_steps=max_episode_steps)
        MountainCarEnv.__init__(self, render_mode=kwargs.pop("render_mode", None),
                                goal_velocity=kwargs.get("goal_velocity", 0))
        self.amplitude = amplitude
        self.gravity_sf = gravity_sf
        self.max_speed_sf = max_speed_sf
        self.force_sf = force_sf


        self.gravity = amplitude*self.gravity*gravity_sf # Effect of changing slope equivalent to changing gravity
        self.max_speed = self.max_speed*max_speed_sf
        self.force = self.force*force_sf
        self.goal_position = goal_position
    
    def _height(self, xs):
        # Override the height function to use a sine wave with scaled amplitude
        return np.sin(3 * xs) * 0.45 * self.amplitude + 0.1 + self.amplitude * 0.45
    
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
        if self.amplitude > 1:
            scale /= self.amplitude
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

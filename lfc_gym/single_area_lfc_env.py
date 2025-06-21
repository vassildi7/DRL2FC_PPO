import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SingleAreaLFCEnv(gym.Env):
    def __init__(
        self,
        max_time=300.0,
        dt=0.1,
        disturbance_segment_duration=20.0,
        disturbance_range=(-0.06, 0.06),
        render_mode=None
    ):
        super().__init__()

        # Simulation parameters
        self.dt = dt
        self.max_steps = int(max_time / dt)
        self.render_mode = render_mode

        # System constants
        self.Tg = 0.2   # Governor time constant
        self.Tt = 0.5   # Turbine time constant
        self.Tp = 20.0  # Power system time constant
        self.Kp = 120.0 # Power system gain
        self.R = 2.4    # Droop characteristic

        # Disturbance configuration
        self.disturbance_segment_duration = disturbance_segment_duration
        self.disturbance_range = disturbance_range
        self.disturbance_schedule = {}
        self.active_dPload = 0.0

        # Action and observation spaces
        self.action_space = spaces.Box(low=np.array([-0.1]), high=np.array([0.1]), dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # 4 states now
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Runtime state variables
        self.t = 0.0
        self.step_count = 0
        self.state = np.zeros(4, dtype=np.float32)  # [Delta_f, Pm, Pv, integral(ACE)]
        self.current_dPload = 0.0
        self.current_dPm = 0.0
        self.disturbance_end_time = 0.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0.0
        self.step_count = 0
        self.state[:] = 0.0
        self.current_dPload = 0.0
        self.current_dPm = 0.0
        self.active_dPload = 0.0
        self.disturbance_end_time = 0.0
        self.disturbance_schedule = {}

        # Schedule disturbances for episode (70% chance)
        if self.np_random.uniform() >= 0.3:
            time_cursor = 30.0
            while time_cursor < self.max_steps * self.dt:
                duration = self.disturbance_segment_duration
                magnitude = self.np_random.uniform(*self.disturbance_range)
                self.disturbance_schedule[round(time_cursor, 1)] = (duration, magnitude)
                time_cursor += duration

        return self.state.astype(np.float32).copy(), {}

    def step(self, action):
        # Clip and convert action
        action = float(np.clip(action, -0.1, 0.1)[0]) if isinstance(action, (list, np.ndarray)) else float(np.clip(action, -0.1, 0.1))

        # Handle disturbances
        if round(self.t, 1) in self.disturbance_schedule:
            duration, magnitude = self.disturbance_schedule[round(self.t, 1)]
            self.active_dPload = magnitude
            self.disturbance_end_time = self.t + duration

        dPload = self.active_dPload if self.t < self.disturbance_end_time else 0.0
        self.current_dPload = dPload

        # System dynamics with integral of ACE as state
        def f_dyn(y):
            f, Pm, Pv, ace_int = y
            dPm_dt = (-Pm + (-f / self.R) + action) / self.Tg
            dPv_dt = (-Pv + Pm) / self.Tt
            df_dt = (self.Kp / self.Tp) * (Pv - dPload) - (f / self.Tp)
            dACEint_dt = f  # ACE assumed equal to frequency deviation in single area
            return np.array([df_dt, dPm_dt, dPv_dt, dACEint_dt], dtype=np.float32)

        # RK4 integration
        y = self.state
        k1 = f_dyn(y)
        k2 = f_dyn(y + self.dt / 2 * k1)
        k3 = f_dyn(y + self.dt / 2 * k2)
        k4 = f_dyn(y + self.dt * k3)
        y_next = y + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state = np.clip(y_next, -1.0, 1.0)
        self.t += self.dt
        self.step_count += 1
        self.current_dPm = y_next[1]

        # Reward calculation
        delta_f = self.state[0]
        ace_int = self.state[3]
        control_effort = action

        freq_penalty = 200000 * delta_f**2
        ace_int_penalty = 10000 * ace_int**2
        effort_penalty = 50 * control_effort**2
        zero_freq_bonus = 200.0 if abs(delta_f) < 0.001 else 0.0
        inactivity_bonus = 50.0 if dPload == 0.0 and abs(control_effort) < 0.001 else 0.0

        reward = - (freq_penalty + ace_int_penalty + effort_penalty) + zero_freq_bonus + inactivity_bonus

        terminated = self.step_count >= self.max_steps
        truncated = False

        return self.state.astype(np.float32).copy(), reward, terminated, truncated, {}

    def render(self):
        print(f"t={self.t:.2f}, Δf={self.state[0]:.4f}, ΔPm={self.state[1]:.4f}, ΔPv={self.state[2]:.4f}, int(ACE)={self.state[3]:.4f}")
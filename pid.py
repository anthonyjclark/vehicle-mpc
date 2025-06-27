import numpy as np
import pandas as pd

from utils import FloatArray
from vehicle import Control, State


class PidController:
    def __init__(self, Kp: float, Ki: float, Kd: float, limits: Pair, dt: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.limits = limits
        self.dt = dt

        self.integral = 0
        self.derivative = 0

        self.prev_value = 0

    def __call__(self, target: float, value: float) -> float:
        min_output, max_output = self.limits

        error = target - value

        proportional = self.Kp * error

        integral = self.integral + self.Ki * error * self.dt

        # Calculate derivative on the measured value instead of the error
        dv = value - self.prev_value
        self.prev_value = value
        self.derivative = -self.Kd * (dv / self.dt)

        output = proportional + integral + self.derivative

        # Only accumulate integral if output is not saturated or error would reduce saturation
        if not ((output >= max_output and error > 0) or (output <= min_output and error < 0)):
            self.integral = integral

        output = np.clip(output, min_output, max_output)
        return output


class PidPathFollower:
    def __init__(
        self,
        reference: pd.DataFrame,
        dt: float,
        Kp_steer: float = 1.0,
        Ki_steer: float = 0.0,
        Kd_steer: float = 0.0,
        max_steer: float = 0.523,
        Kp_accel: float = 1.0,
        Ki_accel: float = 0.0,
        Kd_accel: float = 0.0,
        max_accel: float = 2.0,
    ):
        self.ref_xs: FloatArray = reference["x"].to_numpy(dtype=float)  # type: ignore
        self.ref_ys: FloatArray = reference["y"].to_numpy(dtype=float)  # type: ignore
        self.ref_yaws: FloatArray = reference["yaw"].to_numpy(dtype=float)  # type: ignore
        self.ref_vs: FloatArray = reference["v"].to_numpy(dtype=float)  # type: ignore

        steer_limits = (-max_steer, max_steer)
        accel_limits = (-max_accel, max_accel)

        self.pid_steer = PidController(Kp_steer, Ki_steer, Kd_steer, steer_limits, dt)
        self.pid_accel = PidController(Kp_accel, Ki_accel, Kd_accel, accel_limits, dt)

    def __call__(self, state: State) -> Control:
        x, y, yaw, v = state["x"], state["y"], state["yaw"], state["v"]

        # Find the point on the reference path closest to the vehicle
        dx = self.ref_xs - x
        dy = self.ref_ys - y
        ref_i = np.argmin(np.hypot(dx, dy))

        # Calculate the cross-track error (takes into account the heading and the lateral distance)
        heading_error = self.ref_yaws[ref_i] - yaw
        cross_track_error = np.sin(heading_error) * np.hypot(dx[ref_i], dy[ref_i])
        steer = self.pid_steer.control(cross_track_error)

        # Use velocity error to compute the acceleration
        v_error = self.ref_vs[ref_i] - v
        accel = self.pid_accel.control(v_error)

        return {"steer": steer, "accel": accel}

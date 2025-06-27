import numpy as np
import pandas as pd

from utils import FloatArray
from vehicle import Control, State


class PidController:
    def __init__(self, Kp: float, Ki: float, Kd: float, dt: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0.0
        self.integral = 0.0

    def control(self, error: float) -> float:
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


class PidPathFollower:
    def __init__(
        self,
        reference: pd.DataFrame,
        dt: float,
        Kp_steer: float = 1.0,
        Ki_steer: float = 0.0,
        Kd_steer: float = 0.0,
        Kp_accel: float = 1.0,
        Ki_accel: float = 0.0,
        Kd_accel: float = 0.0,
    ):
        self.ref_xs: FloatArray = reference["x"].to_numpy(dtype=float)  # type: ignore
        self.ref_ys: FloatArray = reference["y"].to_numpy(dtype=float)  # type: ignore
        self.ref_yaws: FloatArray = reference["yaw"].to_numpy(dtype=float)  # type: ignore
        self.ref_vs: FloatArray = reference["v"].to_numpy(dtype=float)  # type: ignore

        self.pid_steer = PidController(Kp_steer, Ki_steer, Kd_steer, dt)
        self.pid_accel = PidController(Kp_accel, Ki_accel, Kd_accel, dt)

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

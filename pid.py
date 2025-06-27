import numpy as np
import pandas as pd

from utils import FloatArray
from vehicle import Control, State, unpack_state

Pair = tuple[float, float]


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


# class PidController:
# def __init__(self, Kp: float, Ki: float, Kd: float, output_max_abs: float, dt: float):
#     self.Kp = Kp
#     self.Ki = Ki
#     self.Kd = Kd
#     self.dt = dt
#     self.prev_error = 0.0
#     self.integral = 0.0
#     self.output_max_abs = output_max_abs

# def __call__(self, error: float) -> float:
#     derivative = (error - self.prev_error) / self.dt

#     temp_integral = self.integral + error * self.dt

#     output = self.Kp * error + self.Ki * temp_integral + self.Kd * derivative
#     output_clipped = np.clip(output, -self.output_max_abs, self.output_max_abs)

#     if (
#         output == output_clipped
#         or (output > output_clipped and error < 0)
#         or (output < output_clipped and error > 0)
#     ):
#         self.integral = temp_integral

#     self.prev_error = error
#     return output_clipped


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
        # x, y, yaw, v = state["x"], state["y"], state["yaw"], state["v"]

        # dxs = self.ref_xs - x
        # dys = self.ref_ys - y
        # ds = np.hypot(dxs, dys)
        # ref_i = np.argmin(ds)

        # k_e = 0.3
        # k_v = 20

        # # 1. Heading error
        # yaw_path = self.ref_yaws[ref_i]
        # yaw_diff = yaw_path - yaw
        # assert -np.pi <= yaw_diff <= np.pi, f"Invalid yaw difference: {yaw_diff}"

        # # 2. Cross-track error
        # dx = dxs[ref_i]
        # dy = dys[ref_i]
        # # crosstrack_error = -np.sin(yaw_path) * dx + np.cos(yaw_path) * dy
        # # print(f"==>> crosstrack_error: {crosstrack_error}")
        # crosstrack_error = np.hypot(dx, dy)
        # # print(f"==>> crosstrack_error: {crosstrack_error}")

        # yaw_cross_track = np.arctan2(-dy, -dx)
        # yaw_path2ct = yaw_path - yaw_cross_track
        # assert -np.pi <= yaw_path2ct <= np.pi, f"Invalid yaw path to cross track: {yaw_path2ct}"

        # crosstrack_error = abs(crosstrack_error) if yaw_path2ct >= 0 else -abs(crosstrack_error)

        # # TODO: why k_v + v
        # yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + v))

        # # 3. Control law
        # steer_expect = yaw_diff + yaw_diff_crosstrack
        # assert -np.pi <= steer_expect <= np.pi, f"Invalid steer expectation: {steer_expect}"
        # steer_expect = min(1.22, steer_expect)
        # steer_expect = max(-1.22, steer_expect)

        # # 4. Update
        # steer_output = steer_expect
        # steer = np.clip(steer_output, -np.deg2rad(30), np.deg2rad(30))

        # # # # Cross-track error (signed)
        # # # cross_track_error = -np.sin(ref_yaw) * map_dx + np.cos(ref_yaw) * map_dy

        # # # Heading error (normalize to [-pi, pi])
        # # heading_error = ref_yaw - yaw
        # # heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        # # assert -np.pi <= heading_error <= np.pi, f"Heading error out of bounds: {heading_error}"

        # # # Stanley control law
        # # k_stanley = 2  # Tune this gain
        # # epsilon = 1e-3  # To avoid division by zero
        # # stanley_term = np.arctan2(k_stanley * cross_track_error, v + epsilon)

        # # steer = heading_error + stanley_term

        # # # Clamp to max steer
        # # steer = np.clip(steer, -np.deg2rad(30), np.deg2rad(30))

        # accel = self.pid_accel(self.ref_vs[ref_i], v)

        # return {"steer": steer, "accel": accel}

        x, y, yaw, v = unpack_state(state)

        # Find the point on the reference path closest to the vehicle
        dxs = self.ref_xs - x
        dys = self.ref_ys - y
        ds = np.hypot(dxs, dys)
        ref_i = np.argmin(ds)

        ds = ds[ref_i]
        # TODO: add 4 as a parameter
        if ds > 4:
            raise ValueError(f"Vehicle is too far from the reference path: {ds:.2f} m. ")

        # TODO: figure out cross-track error
        # Calculate the cross-track error (takes into account the heading and the lateral distance)
        # cross_track_error = np.sin(heading_error) * np.hypot(dx[ref_i], dy[ref_i])

        dxs = dxs[ref_i]
        dys = dys[ref_i]

        # cross_track_error = np.sin(self.ref_yaws[ref_i] - yaw) * np.hypot(dx, dy)
        # cross_track_error = ( -np.sin(self.ref_yaws[ref_i]) * dx + np.cos(self.ref_yaws[ref_i]) * dy )
        cross_track_error = 0

        # TODO: normalize the heading error
        heading_error = -(self.ref_yaws[ref_i] - yaw)

        steer_input = cross_track_error + heading_error

        steer = self.pid_steer(0.0, steer_input)

        accel = self.pid_accel(self.ref_vs[ref_i], v)

        return {"steer": steer, "accel": accel}

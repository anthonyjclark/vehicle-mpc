from typing import TypedDict

import numpy as np


class Control(TypedDict):
    steer: float  # front tire angle of the vehicle [rad] (counterclockwize)
    accel: float  # longitudinal acceleration of the vehicle [m/s^2]


class State(TypedDict):
    x: float  # x-axis position in the global frame [m]
    y: float  # y-axis position in the global frame [m]
    yaw: float  # orientation in the global frame [rad]
    v: float  # longitudinal velocity [m/s]


def unpack_state(state: State) -> tuple[float, float, float, float]:
    return state["x"], state["y"], state["yaw"], state["v"]


def unpack_control(control: Control) -> tuple[float, float]:
    return control["steer"], control["accel"]


class Vehicle:
    "Dynamics follow that of the bicycle model."

    def __init__(
        self,
        init_state: State | None = None,
        wheel_base: float = 2.5,  # [m]
        max_steer_abs: float = 0.523,  # [rad]
        max_accel_abs: float = 2.000,  # [m/s^2]
        dt: float = 0.05,  # [s]
    ):
        self.wheel_base = wheel_base
        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.dt = dt

        self.reset(init_state)

    def reset(self, init_state: State | None = None) -> None:
        self.state = (
            init_state
            if (init_state is not None)
            else {"x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0}
        )

    def update(self, u: Control) -> State:
        x, y, yaw, v = unpack_state(self.state)

        steer = np.clip(u["steer"], -self.max_steer_abs, self.max_steer_abs)
        accel = np.clip(u["accel"], -self.max_accel_abs, self.max_accel_abs)

        self.state: State = {
            "x": x + v * np.cos(yaw) * self.dt,
            "y": y + v * np.sin(yaw) * self.dt,
            "yaw": yaw + v / self.wheel_base * np.tan(steer) * self.dt,
            "v": v + accel * self.dt,
        }

        return self.state


if __name__ == "__main__":
    from utils import stadium_trajectory

    reference = stadium_trajectory()
    vehicle = Vehicle()

    num_steps = 100
    for i in range(num_steps):
        steer = 0.6 * np.sin(i / 3.0)
        accel = 2.2 * np.sin(i / 10.0)
        vehicle.update(u={"steer": steer, "accel": accel})

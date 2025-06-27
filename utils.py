# TODO: need to cleanup animation creation code
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from matplotlib.animation import Animation, ArtistAnimation, HTMLWriter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm.auto import tqdm

from vehicle import Control, State, unpack_control, unpack_state

FloatArray = NDArray[np.float_]


def circular_trajectory(
    count: int = 200, radius: float = 20.0, speed: float = 1.0
) -> DataFrame:
    angles = np.linspace(0, 2 * np.pi, count)
    return DataFrame(
        {
            "x": radius * np.cos(angles),
            "y": radius * np.sin(angles),
            "yaw": angles + np.pi / 2.0,
            "v": [speed] * count,
        }
    )


def stadium_trajectory(
    count: int = 200, radius: float = 20.0, length: float = 50.0, speed: float = 1.0
) -> DataFrame:
    straight_length = length * 2
    curve_length = 2 * np.pi * radius
    total_length = straight_length + curve_length

    straight_count = int(count * straight_length / total_length)
    curve_count = count - straight_count

    mid_cu = curve_count // 2
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, curve_count)
    x_cu = radius * np.cos(angles)
    y_cu = radius * np.sin(angles)
    yaw_cu = angles + np.pi

    half_cu = straight_count // 2
    x_st = np.linspace(0, length, half_cu)
    y_st = np.array([radius] * half_cu)
    yaw_st = np.array([0.0] * half_cu)

    xs = np.concatenate((x_st, x_cu[:mid_cu] + length, x_st[::-1], x_cu[mid_cu:]))
    ys = np.concatenate((-y_st, y_cu[:mid_cu], y_st, y_cu[mid_cu:]))
    yaws = np.concatenate((yaw_st, yaw_cu[:mid_cu], yaw_st + np.pi, yaw_cu[mid_cu:]))

    return DataFrame({"x": xs, "y": ys, "yaw": yaws, "v": [speed] * count})


def to_jshtml(
    animation: Animation,
    fps: int,
    progress_callback: Callable[[int, int], None],
    embed_frames: bool = True,
    default_mode: str = "loop",
) -> str:
    """
    Adding a progress callback to the to_jshtml function
    Starting with existing version from:
    https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/animation.py#L1337-L1379"

    TODO: submit a PR
    """

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir, "temp.html")
        writer = HTMLWriter(fps=fps, embed_frames=embed_frames, default_mode=default_mode)
        animation.save(str(path), writer=writer, progress_callback=progress_callback)
        html_representation: str = path.read_text()

    return html_representation


def save_animation(
    reference: DataFrame,
    states: DataFrame,
    controls: DataFrame,
    max_steer: float,
    max_accel: float,
    interval_ms: int,
    filename: str,
    movie_writer: str = "ffmpeg",
) -> None:
    animation, _ = create_animation(
        reference,
        states,
        controls,
        max_steer,
        max_accel,
        interval_ms,
    )
    animation.save(filename, writer=movie_writer)


def show_animation(
    reference: DataFrame,
    states: DataFrame,
    controls: DataFrame,
    max_steer: float,
    max_accel: float,
    interval_ms: int,
) -> None:
    def progress_callback(current: int, total: int) -> None:
        to_jshtml_pbar.n = current + 1
        to_jshtml_pbar.refresh()
        if current + 1 == total:
            to_jshtml_pbar.close()

    animation, length = create_animation(
        reference,
        states,
        controls,
        max_steer,
        max_accel,
        interval_ms,
    )

    # Use the custom to_jshtml method so that we can show progress
    to_jshtml_pbar = tqdm(total=length)
    jshtml = to_jshtml(animation, 1000 // interval_ms, progress_callback)
    html = display.HTML(jshtml)
    display.display(html)
    plt.close()


def create_animation(
    reference: DataFrame,
    states: DataFrame,
    controls: DataFrame,
    max_steer: float,
    max_accel: float,
    interval_ms: int,
    vehicle_width: float = 2.00,
    vehicle_length: float = 2.80,
    wheel_width: float = 0.4,
    wheel_length: float = 0.9,
):
    view_x_lim_min, view_x_lim_max = -20.0, 20.0
    view_y_lim_min, view_y_lim_max = -25.0, 25.0

    figure: Figure = plt.figure(figsize=(9, 9))
    figure.tight_layout()

    ax_main: Axes = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3)
    ax_main.set_aspect("equal")
    ax_main.set_xlim(view_x_lim_min, view_x_lim_max)
    ax_main.set_ylim(view_y_lim_min, view_y_lim_max)
    ax_main.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax_main.tick_params(bottom=False, left=False, right=False, top=False)

    ax_mini: Axes = plt.subplot2grid((3, 4), (0, 3))
    ax_mini.set_aspect("equal")
    ax_mini.axis("off")

    ax_steer: Axes = plt.subplot2grid((3, 4), (1, 3))
    ax_steer.set_title("Steering Angle", fontsize="12")
    ax_steer.axis("off")

    ax_accel: Axes = plt.subplot2grid((3, 4), (2, 3))
    ax_accel.set_title("Acceleration", fontsize="12")
    ax_accel.axis("off")

    state_iter = states.iterrows()
    control_iter = controls.iterrows()
    frames: list[list[Artist]] = []

    for (_, state), (_, control) in tqdm(zip(state_iter, control_iter), total=len(states)):
        state = dict(state)
        control = dict(control)
        frame = create_frame(
            reference=reference,
            state=state,  # type: ignore
            control=control,  # type: ignore
            vehicle_width=vehicle_width,
            vehicle_length=vehicle_length,
            wheel_width=wheel_width,
            wheel_length=wheel_length,
            max_steer=max_steer,
            max_accel=max_accel,
            ax_main=ax_main,
            ax_mini=ax_mini,
            ax_steer=ax_steer,
            ax_accel=ax_accel,
        )
        frames.append(frame)

    return ArtistAnimation(figure, frames, interval=interval_ms), len(frames)


def create_frame(
    reference: DataFrame,
    state: State,
    control: Control,
    vehicle_width: float,
    vehicle_length: float,
    wheel_width: float,
    wheel_length: float,
    max_steer: float,
    max_accel: float,
    ax_main: Axes,
    ax_mini: Axes,
    ax_steer: Axes,
    ax_accel: Axes,
    optimal_traj: FloatArray | None = None,
    sampled_trajs: FloatArray | None = None,
) -> list[Artist | Line2D | Polygon]:
    x, y, yaw, v = unpack_state(state)
    steer, accel = unpack_control(control)

    #
    # Draw the vehicle
    #

    xs = np.array([-0.5, -0.5, 0.5, 0.5, -0.5]) * vehicle_length
    ys = np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * vehicle_width
    xs, ys = affine_transform(xs, ys, angle=yaw)
    frame = ax_main.plot(xs, ys, color="black", linewidth=2.0, zorder=3)

    # Center of the vehicle
    r = vehicle_width / 20.0
    center = Circle((0, 0), radius=r, fc="white", ec="black", linewidth=2.0, zorder=6)
    frame += [ax_main.add_artist(center)]

    #
    # Draw the vehicle wheels
    #

    vl, vw = vehicle_length, vehicle_width

    xs = np.array([-0.5, -0.5, 0.5, 0.5]) * wheel_length
    ys = np.array([-0.5, 0.5, 0.5, -0.5]) * wheel_width

    # Rear-left wheel
    xs_xform, ys_xform = affine_transform(xs, ys, x=-0.3 * vl, y=0.3 * vw)
    xs_xform, ys_xform = affine_transform(xs_xform, ys_xform, angle=yaw)
    frame += ax_main.fill(xs_xform, ys_xform, color="black", zorder=3)

    # Rear-right wheel
    xs_xform, ys_xform = affine_transform(xs, ys, x=-0.3 * vl, y=-0.3 * vw)
    xs_xform, ys_xform = affine_transform(xs_xform, ys_xform, angle=yaw)
    frame += ax_main.fill(xs_xform, ys_xform, color="black", zorder=3)

    # Front-left wheel
    xs_xform, ys_xform = affine_transform(xs, ys, steer, x=0.3 * vl, y=0.3 * vw)
    xs_xform, ys_xform = affine_transform(xs_xform, ys_xform, angle=yaw)
    frame += ax_main.fill(xs_xform, ys_xform, color="black", zorder=3)

    ## Front-right wheel
    xs_xform, ys_xform = affine_transform(xs, ys, steer, x=0.3 * vl, y=-0.3 * vw)
    xs_xform, ys_xform = affine_transform(xs_xform, ys_xform, angle=yaw)
    frame += ax_main.fill(xs_xform, ys_xform, color="black", zorder=3)

    ref_x = reference["x"] - np.full(reference.shape[0], x)
    ref_y = reference["y"] - np.full(reference.shape[0], y)
    frame += ax_main.plot(ref_x, ref_y, color="black", linestyle="dashed", linewidth=1.5)

    #
    # Vehicle diagnostics
    #

    # TODO: consider pose information
    text = f"Speed = {v:>+3.1f} [m/s]"
    frame += [
        ax_main.text(
            0.5,
            0.02,
            text,
            ha="center",
            transform=ax_main.transAxes,
            fontsize=14,
            fontfamily="monospace",
        )
    ]

    #
    # Minimap
    #

    xs = np.array([-0.5, -0.5, 0.5, 0.5, -0.5]) * vehicle_length
    ys = np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * vehicle_width

    frame += ax_mini.plot(reference["x"], reference["y"], color="black", linestyle="dashed")
    xs, ys = affine_transform(xs, ys, yaw, x=x, y=y)
    frame += ax_mini.plot(xs, ys, color="black", linewidth=2.0, zorder=3)
    frame += ax_mini.fill(xs, ys, color="white", zorder=2)

    #
    # Steering display
    #

    wedge_max = 3.0 / 4.0
    wedge_start = 225
    s_abs = np.abs(steer)

    left_adjust = 0.0 if steer < 0.0 else -s_abs
    right_adjust = -s_abs if steer < 0.0 else 0.0

    wedges = [
        (max_steer + left_adjust) * wedge_max,
        s_abs * wedge_max,
        (max_steer + right_adjust) * wedge_max,
        2 * max_steer * (1 - wedge_max),
    ]

    steer_pie_obj, _ = ax_steer.pie(  # type: ignore
        wedges,
        startangle=wedge_start,
        counterclock=False,
        colors=["lightgray", "black", "lightgray", "white"],
        wedgeprops={"linewidth": 0, "edgecolor": "white", "width": 0.4},
    )
    frame += steer_pie_obj

    frame += [
        ax_steer.text(
            0,
            -1,
            f"{np.rad2deg(steer):+.2f} [deg]",
            size=14,
            horizontalalignment="center",
            verticalalignment="center",
            fontfamily="monospace",
        )
    ]

    #
    # Trajectory samples
    #

    # # draw the predicted optimal trajectory from mppi
    # if optimal_traj and optimal_traj.any():
    #     optimal_traj_x_offset = np.ravel(optimal_traj[:, 0]) - np.full(optimal_traj.shape[0], x)
    #     optimal_traj_y_offset = np.ravel(optimal_traj[:, 1]) - np.full(optimal_traj.shape[0], y)
    #     frame += ax_main.plot(
    #         optimal_traj_x_offset,
    #         optimal_traj_y_offset,
    #         color="#990099",
    #         linestyle="solid",
    #         linewidth=2.0,
    #         zorder=5,
    #     )

    # # draw the sampled trajectories from mppi
    # if sampled_trajs and sampled_trajs.any():
    #     min_alpha_value = 0.25
    #     max_alpha_value = 0.35
    #     for idx, sampled_traj in enumerate(sampled_trajs):
    #         # draw darker for better samples
    #         alpha_value = (1.0 - (idx + 1) / len(sampled_trajs)) * (
    #             max_alpha_value - min_alpha_value
    #         ) + min_alpha_value
    #         sampled_traj_x_offset = np.ravel(sampled_traj[:, 0]) - np.full(
    #             sampled_traj.shape[0], x
    #         )
    #         sampled_traj_y_offset = np.ravel(sampled_traj[:, 1]) - np.full(
    #             sampled_traj.shape[0], y
    #         )
    #         frame += ax_main.plot(
    #             sampled_traj_x_offset,
    #             sampled_traj_y_offset,
    #             color="gray",
    #             linestyle="solid",
    #             linewidth=0.2,
    #             zorder=4,
    #             alpha=alpha_value,
    #         )

    # # acceleration
    # pie_rate = 3.0 / 4.0
    # pie_start = 225
    # a_abs = np.abs(accel)
    # if accel > 0.0:
    #     accel_pie_obj, _ = ax_accel.pie(  # type: ignore
    #         [
    #             max_accel * pie_rate,
    #             a_abs * pie_rate,
    #             (max_accel - a_abs) * pie_rate,
    #             2 * max_accel * (1 - pie_rate),
    #         ],
    #         startangle=pie_start,
    #         counterclock=False,
    #         colors=["lightgray", "black", "lightgray", "white"],
    #         wedgeprops={"linewidth": 0, "edgecolor": "white", "width": 0.4},
    #     )
    # else:
    #     accel_pie_obj, _ = ax_accel.pie(  # type: ignore
    #         [
    #             (max_accel - a_abs) * pie_rate,
    #             a_abs * pie_rate,
    #             max_accel * pie_rate,
    #             2 * max_accel * (1 - pie_rate),
    #         ],
    #         startangle=pie_start,
    #         counterclock=False,
    #         colors=["lightgray", "black", "lightgray", "white"],
    #         wedgeprops={"linewidth": 0, "edgecolor": "white", "width": 0.4},
    #     )
    # frame += accel_pie_obj
    # frame += [
    #     ax_accel.text(
    #         0,
    #         -1,
    #         f"{accel:+.2f} " + r"$ \rm{[m/s^2]}$",
    #         size=14,
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #         fontfamily="monospace",
    #     )
    # ]

    return frame


def affine_transform(
    xlist: list[float] | FloatArray,
    ylist: list[float] | FloatArray,
    angle: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
) -> tuple[list[float], list[float]]:
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    x_xform = [xval * cos_angle - yval * sin_angle + x for xval, yval in zip(xlist, ylist)]
    y_xform = [xval * sin_angle + yval * cos_angle + y for xval, yval in zip(xlist, ylist)]
    return x_xform, y_xform

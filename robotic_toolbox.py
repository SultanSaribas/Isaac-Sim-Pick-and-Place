"""
StableFrankaPickPlaceController
-------------------------------

This file keeps the old class name:
    RoboticsToolboxPickPlaceController

But this version DOES NOT use Robotics Toolbox IK.

Reason:
    RTB Panda model and Isaac Sim Franka articulation/end-effector frames do not
    match perfectly. Directly applying RTB IK joint targets can cause shaking,
    branch flips, self-collision, and unstable motion.

This version uses Isaac Sim's own Franka KinematicsSolver.
It also avoids diagonal transfer through the robot base by using safer L-shaped
waypoints during transfer phases.
"""

import logging
from typing import List, Optional

import numpy as np

from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction

try:
    from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
except ImportError:
    from omni.isaac.franka import KinematicsSolver


log = logging.getLogger("stable_franka_controller")


class RoboticsToolboxPickPlaceController:
    """
    Stable pick-and-place controller for Isaac Sim Franka.

    Phases:
      0 - Move EE above pick
      1 - Lower EE to pick
      2 - Wait
      3 - Close gripper
      4 - Lift object
      5 - Move above place with safe L-shaped transfer
      6 - Lower to place
      7 - Open gripper
      8 - Lift after release
      9 - Return with safe L-shaped transfer
    """

    _MOTION_PHASES = frozenset({0, 1, 4, 5, 6, 8, 9})

    def __init__(
        self,
        name: str,
        gripper,
        robot_articulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        self._name = name
        self._gripper = gripper
        self._robot_articulation = robot_articulation

        self._kinematics_solver = KinematicsSolver(
            robot_articulation=self._robot_articulation
        )

        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.45 / get_stage_units()

        self._safe_transfer_height = 0.55 / get_stage_units()

        self._h0: Optional[float] = None
        self._pick_xy: Optional[np.ndarray] = None

        self._event = 0
        self._phase_step = 0
        self._phase_n_steps: Optional[int] = None

        self._phase_start_position: Optional[np.ndarray] = None
        self._phase_target_position: Optional[np.ndarray] = None
        self._phase_waypoints: Optional[List[np.ndarray]] = None

        self._last_ik_success = True
        self._pause = False

        if events_dt is None:
            self._events_dt: List[float] = [
                0.008,   # 0 - move above pick
                0.005,   # 1 - lower to pick
                1.0,     # 2 - wait
                0.02,    # 3 - close gripper
                0.01,    # 4 - lift object
                0.005,   # 5 - safe transfer to place
                0.01,    # 6 - lower to place
                1.0,     # 7 - open gripper
                0.008,   # 8 - lift after release
                0.005,   # 9 - safe return
            ]
        else:
            if len(events_dt) != 10:
                raise ValueError("events_dt must have exactly 10 elements.")
            self._events_dt = list(events_dt)

        log.info(
            "[%s] Stable Isaac IK controller created | h1=%.4f | safe_z=%.4f | events_dt=%s",
            self._name,
            self._h1,
            self._safe_transfer_height,
            self._events_dt,
        )

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

    def is_paused(self) -> bool:
        return self._pause

    def get_current_event(self) -> int:
        return self._event

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False

    def reset(self) -> None:
        self._event = 0
        self._phase_step = 0
        self._phase_n_steps = None

        self._phase_start_position = None
        self._phase_target_position = None
        self._phase_waypoints = None

        self._h0 = None
        self._pick_xy = None

        self._last_ik_success = True
        self._pause = False

        log.info("[%s] Controller reset", self._name)

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.zeros(3)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(
                np.array([0.0, np.pi, 0.0])
            )

        if self._pause or self.is_done():
            self.pause()
            return ArticulationAction(
                joint_positions=[None] * current_joint_positions.shape[0]
            )

        event = self._event
        n_joints = current_joint_positions.shape[0]

        # ---------------------------------------------------------------
        # Initialize phase
        # ---------------------------------------------------------------
        if self._phase_n_steps is None:
            self._phase_n_steps = max(1, int(round(1.0 / self._events_dt[event])))
            self._last_ik_success = True

            if event == 0:
                self._pick_xy = np.array(
                    [
                        float(picking_position[0]),
                        float(picking_position[1]),
                    ]
                )
                self._h0 = float(picking_position[2])

                log.info(
                    "[%s] Pick initialized | pick_xy=%s | h0=%.4f | h1=%.4f",
                    self._name,
                    np.round(self._pick_xy, 4),
                    self._h0,
                    self._h1,
                )

            if event in self._MOTION_PHASES:
                self._phase_start_position = self._get_current_ee_position()
                self._phase_target_position = self._phase_target(
                    event=event,
                    placing_position=placing_position,
                    offset=end_effector_offset,
                )

                self._phase_waypoints = self._build_safe_waypoints(
                    event=event,
                    start=self._phase_start_position,
                    target=self._phase_target_position,
                )

                log.info(
                    "[%s] Phase %d | start=%s | target=%s | steps=%d | waypoints=%s",
                    self._name,
                    event,
                    np.round(self._phase_start_position, 4),
                    np.round(self._phase_target_position, 4),
                    self._phase_n_steps,
                    [np.round(p, 4).tolist() for p in self._phase_waypoints],
                )

        # ---------------------------------------------------------------
        # Execute phase
        # ---------------------------------------------------------------
        if event == 2:
            action = ArticulationAction(joint_positions=[None] * n_joints)

        elif event == 3:
            action = self._gripper.forward(action="close")

        elif event == 7:
            action = self._gripper.forward(action="open")

        else:
            target_position = self._interpolated_target_position()

            gripper_state = None
            if event in (4, 5, 6):
                gripper_state = "close"
            elif event in (8, 9):
                gripper_state = "open"

            action = self._compute_ik_action(
                target_position=target_position,
                target_orientation=end_effector_orientation,
                current_joint_positions=current_joint_positions,
                gripper_state=gripper_state,
            )

        # ---------------------------------------------------------------
        # Do not advance if IK failed
        # ---------------------------------------------------------------
        if event in self._MOTION_PHASES and not self._last_ik_success:
            return action

        # ---------------------------------------------------------------
        # Advance phase
        # ---------------------------------------------------------------
        self._phase_step += 1

        if self._phase_step >= self._phase_n_steps:
            log.info("[%s] Phase %d complete", self._name, event)

            self._event += 1
            self._phase_step = 0
            self._phase_n_steps = None

            self._phase_start_position = None
            self._phase_target_position = None
            self._phase_waypoints = None

            self._last_ik_success = True

            if self.is_done():
                log.info("[%s] Pick and place complete", self._name)

        return action

    def _get_current_ee_position(self) -> np.ndarray:
        position, _ = self._kinematics_solver.compute_end_effector_pose(
            position_only=True
        )
        return np.array(position, dtype=float)

    def _interpolated_target_position(self) -> np.ndarray:
        if self._phase_waypoints is None or len(self._phase_waypoints) < 2:
            raise RuntimeError("Motion phase waypoints were not initialized.")

        if self._phase_n_steps <= 1:
            alpha = 1.0
        else:
            alpha = self._phase_step / float(self._phase_n_steps - 1)

        alpha = float(np.clip(alpha, 0.0, 1.0))

        waypoints = self._phase_waypoints
        n_segments = len(waypoints) - 1

        scaled = alpha * n_segments
        seg_idx = min(int(np.floor(scaled)), n_segments - 1)
        local_alpha = scaled - seg_idx

        p0 = waypoints[seg_idx]
        p1 = waypoints[seg_idx + 1]

        return (1.0 - local_alpha) * p0 + local_alpha * p1

    def _build_safe_waypoints(
        self,
        event: int,
        start: np.ndarray,
        target: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Build safer Cartesian waypoints.

        Direct motion from [0.3, 0.3] to [-0.3, -0.3] passes close to the robot base.
        That can cause self-collision or IK failure.

        For transfer phases, use an L-shaped route around the base.
        """
        start = np.array(start, dtype=float)
        target = np.array(target, dtype=float)

        safe_z = max(
            float(start[2]),
            float(target[2]),
            float(self._safe_transfer_height),
        )

        if event == 5:
            return [
                start,
                np.array([start[0], start[1], safe_z]),
                np.array([start[0], target[1], safe_z]),
                np.array([target[0], target[1], safe_z]),
                target,
            ]

        if event == 9:
            return [
                start,
                np.array([start[0], start[1], safe_z]),
                np.array([target[0], start[1], safe_z]),
                np.array([target[0], target[1], safe_z]),
                target,
            ]

        return [
            start,
            target,
        ]

    def _phase_target(
        self,
        event: int,
        placing_position: np.ndarray,
        offset: np.ndarray,
    ) -> np.ndarray:
        if self._pick_xy is None or self._h0 is None:
            raise RuntimeError("Pick pose is not initialized.")

        px = self._pick_xy[0] + offset[0]
        py = self._pick_xy[1] + offset[1]

        h0 = self._h0 + offset[2]
        h1 = self._h1 + offset[2]

        gx = float(placing_position[0]) + offset[0]
        gy = float(placing_position[1]) + offset[1]
        gh = float(placing_position[2]) + offset[2]

        targets = {
            0: np.array([px, py, h1]),
            1: np.array([px, py, h0]),
            4: np.array([px, py, h1]),
            5: np.array([gx, gy, h1]),
            6: np.array([gx, gy, gh]),
            8: np.array([gx, gy, h1]),
            9: np.array([px, py, h1]),
        }

        if event not in targets:
            raise ValueError(f"No target pose defined for event {event}")

        return targets[event]

    def _compute_ik_action(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        current_joint_positions: np.ndarray,
        gripper_state: Optional[str] = None,
    ) -> ArticulationAction:
        self._last_ik_success = True

        ik_action = None
        success = False

        # First try IK with orientation.
        try:
            ik_action, success = self._kinematics_solver.compute_inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                position_tolerance=0.015,
                orientation_tolerance=0.4,
            )
        except Exception as exc:
            log.warning(
                "[%s] IK with orientation raised exception: %s",
                self._name,
                exc,
            )

        # If orientation IK fails, try position-only IK.
        if not success:
            try:
                ik_action, success = self._kinematics_solver.compute_inverse_kinematics(
                    target_position=target_position,
                    target_orientation=None,
                    position_tolerance=0.02,
                )
            except Exception as exc:
                log.warning(
                    "[%s] Position-only IK raised exception: %s",
                    self._name,
                    exc,
                )

        if not success:
            self._last_ik_success = False

            log.warning(
                "[%s] IK failed | phase=%d | step=%d | target=%s | holding joints",
                self._name,
                self._event,
                self._phase_step,
                np.round(target_position, 4),
            )

            full_joint_positions = list(current_joint_positions.copy().astype(float))

            if gripper_state == "close" and len(full_joint_positions) >= 9:
                full_joint_positions[-2:] = list(self._gripper.joint_closed_positions)
            elif gripper_state == "open" and len(full_joint_positions) >= 9:
                full_joint_positions[-2:] = list(self._gripper.joint_opened_positions)

            return ArticulationAction(joint_positions=full_joint_positions)

        return self._merge_ik_and_gripper_action(
            ik_action=ik_action,
            current_joint_positions=current_joint_positions,
            gripper_state=gripper_state,
        )

    def _merge_ik_and_gripper_action(
        self,
        ik_action: ArticulationAction,
        current_joint_positions: np.ndarray,
        gripper_state: Optional[str],
    ) -> ArticulationAction:
        full_joint_positions = list(current_joint_positions.copy().astype(float))

        ik_joint_positions = getattr(ik_action, "joint_positions", None)
        ik_joint_indices = getattr(ik_action, "joint_indices", None)

        if ik_joint_positions is not None:
            ik_joint_positions = np.array(ik_joint_positions, dtype=float)

            if ik_joint_indices is None or len(ik_joint_indices) == 0:
                count = min(len(ik_joint_positions), len(full_joint_positions))
                for i in range(count):
                    full_joint_positions[i] = float(ik_joint_positions[i])
            else:
                for joint_index, joint_value in zip(ik_joint_indices, ik_joint_positions):
                    joint_index = int(joint_index)
                    if 0 <= joint_index < len(full_joint_positions):
                        full_joint_positions[joint_index] = float(joint_value)

        if gripper_state == "close" and len(full_joint_positions) >= 9:
            full_joint_positions[-2:] = list(self._gripper.joint_closed_positions)

        elif gripper_state == "open" and len(full_joint_positions) >= 9:
            full_joint_positions[-2:] = list(self._gripper.joint_opened_positions)

        return ArticulationAction(joint_positions=full_joint_positions)

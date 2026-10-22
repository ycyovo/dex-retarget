from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Sequence
from typing import Union

import numpy as np
import yaml
import os

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.yourdfpy import DUMMY_JOINT_NAMES


@dataclass
class RetargetingConfig:
    type: str
    urdf_path: str

    # Whether to add free joint to the root of the robot. Free joint enable the robot hand move freely in the space
    add_dummy_free_joint: bool = False

    # Source refers to the retargeting input, which usually corresponds to the human hand
    # The joint indices of human hand joint which corresponds to each link in the target_link_names
    target_link_human_indices: Optional[np.ndarray] = None

    # The link on the robot hand which corresponding to the wrist of human hand
    wrist_link_name: Optional[str] = None

    # Position retargeting link names
    target_link_names: Optional[List[str]] = None

    # Vector retargeting link names
    target_joint_names: Optional[List[str]] = None
    target_origin_link_names: Optional[List[str]] = None
    target_task_link_names: Optional[List[str]] = None

    # DexPilot retargeting link names
    finger_tip_link_names: Optional[List[str]] = None

    # Scaling factor for vector retargeting only
    # For example, Allegro is 1.6 times larger than normal human hand, then this scaling factor should be 1.6
    scaling_factor: float = 1.0

    # Optimization parameters
    normal_delta: float = 4e-3
    huber_delta: float = 2e-2

    # DexPilot optimizer parameters
    project_dist: float = 0.03
    escape_dist: float = 0.05

    # Optional DexPilot bone-angle loss. Zero preserves the existing objective.
    joint_angle_weight: float = 0.0
    joint_angle_ranges: Optional[Dict[str, float]] = None
    joint_angle_mapping: Optional[Dict[str, Dict[str, Any]]] = None

    # Hard affine coupling: follower = multiplier * source + offset (radians).
    # Followers are removed from optimization variables, but retained in output.
    joint_couplings: Optional[Dict[str, Dict[str, Any]]] = None

    # Joint limit tag
    has_joint_limits: bool = True

    # Optional per-joint limits in radians. These may only tighten URDF limits.
    joint_limits: Optional[Dict[str, Sequence[float]]] = None

    # Mimic joint tag
    ignore_mimic_joint: bool = False

    # Low pass filter
    low_pass_alpha: float = 0.1

    _TYPE = ["vector", "position", "dexpilot"]
    _DEFAULT_URDF_DIR = "./"

    def __post_init__(self):
        # Retargeting type check
        self.type = self.type.lower()
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")
        if not np.isfinite(self.joint_angle_weight) or self.joint_angle_weight < 0:
            raise ValueError("joint_angle_weight must be finite and nonnegative")
        if self.joint_angle_weight > 0 and self.type != "dexpilot":
            raise ValueError("joint_angle_weight is supported only for DexPilot")

        # Vector retargeting requires: target_origin_link_names + target_task_link_names
        # Position retargeting requires: target_link_names
        if self.type == "vector":
            if (
                self.target_origin_link_names is None
                or self.target_task_link_names is None
            ):
                raise ValueError(
                    "Vector retargeting requires: target_origin_link_names + target_task_link_names"
                )
            if len(self.target_task_link_names) != len(self.target_origin_link_names):
                raise ValueError(
                    "Vector retargeting origin and task links dim mismatch"
                )
            if self.target_link_human_indices.shape != (
                2,
                len(self.target_origin_link_names),
            ):
                raise ValueError(
                    "Vector retargeting link names and link indices dim mismatch"
                )
            if self.target_link_human_indices is None:
                raise ValueError(
                    "Vector retargeting requires: target_link_human_indices"
                )

        elif self.type == "position":
            if self.target_link_names is None:
                raise ValueError("Position retargeting requires: target_link_names")
            self.target_link_human_indices = self.target_link_human_indices.squeeze()
            if self.target_link_human_indices.shape != (len(self.target_link_names),):
                raise ValueError(
                    "Position retargeting link names and link indices dim mismatch"
                )
            if self.target_link_human_indices is None:
                raise ValueError(
                    "Position retargeting requires: target_link_human_indices"
                )

        elif self.type == "dexpilot":
            if self.finger_tip_link_names is None or self.wrist_link_name is None:
                raise ValueError(
                    "Position retargeting requires: finger_tip_link_names + wrist_link_name"
                )
            if self.target_link_human_indices is not None:
                print(
                    "\033[33m",
                    "Target link human indices is provided in the DexPilot retargeting config, which is uncommon.\n"
                    "If you do not know exactly how it is used, please leave it to None for default.\n"
                    "\033[00m",
                )

        # URDF path check
        urdf_path = Path(self.urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = self._DEFAULT_URDF_DIR / urdf_path
            urdf_path = urdf_path.absolute()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.urdf_path = str(urdf_path)

    @classmethod
    def set_default_urdf_dir(cls, urdf_dir: Union[str, Path]):
        path = Path(urdf_dir)
        if not path.exists():
            raise ValueError(f"URDF dir {urdf_dir} not exists.")
        cls._DEFAULT_URDF_DIR = urdf_dir

    @classmethod
    def load_from_file(
        cls, config_path: Union[str, Path], override: Optional[Dict] = None
    ):
        path = Path(config_path)
        if not path.is_absolute():
            path = path.absolute()

        with path.open("r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yaml_config["retargeting"]
            return cls.from_dict(cfg, override)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], override: Optional[Dict] = None):
        if "target_link_human_indices" in cfg:
            cfg["target_link_human_indices"] = np.array(
                cfg["target_link_human_indices"]
            )
        if override is not None:
            for key, value in override.items():
                cfg[key] = value
        config = RetargetingConfig(**cfg)
        return config

    def build(self) -> SeqRetargeting:
        from dex_retargeting.optimizer import (
            VectorOptimizer,
            PositionOptimizer,
            DexPilotOptimizer,
        )
        import tempfile

        # Process the URDF with yourdfpy to better find file path
        robot_urdf = urdf.URDF.load(
            self.urdf_path,
            add_dummy_free_joints=self.add_dummy_free_joint,
            build_scene_graph=False,
        )
        urdf_name = self.urdf_path.split(os.path.sep)[-1]
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)

        # Load pinocchio model
        robot = RobotWrapper(temp_path)

        effective_joint_limits = robot.joint_limits.copy()
        if self.joint_limits is not None:
            if not self.has_joint_limits:
                raise ValueError(
                    "joint_limits cannot be used when has_joint_limits is false"
                )
            robot_joint_names = robot.dof_joint_names
            for joint_name, limits in self.joint_limits.items():
                if joint_name not in robot_joint_names:
                    raise ValueError(
                        f"joint_limits contains unknown joint: {joint_name}"
                    )
                values = np.asarray(limits, dtype=np.float64)
                if values.shape != (2,) or not np.isfinite(values).all():
                    raise ValueError(
                        f"joint limit for {joint_name} must be two finite values"
                    )
                lower, upper = values
                index = robot_joint_names.index(joint_name)
                urdf_lower, urdf_upper = effective_joint_limits[index]
                if lower > upper:
                    raise ValueError(f"joint limit for {joint_name} must be ordered")
                if lower < urdf_lower or upper > urdf_upper:
                    raise ValueError(
                        f"joint limit for {joint_name} must be within URDF range "
                        f"[{urdf_lower}, {urdf_upper}]"
                    )
                effective_joint_limits[index] = (lower, upper)

        # Add 6D dummy joint to target joint names so that it will also be optimized
        if self.add_dummy_free_joint and self.target_joint_names is not None:
            self.target_joint_names = DUMMY_JOINT_NAMES + self.target_joint_names
        joint_names = (
            self.target_joint_names
            if self.target_joint_names is not None
            else robot.dof_joint_names
        )

        # Merge explicit hardware couplings with the existing URDF mimic model.
        _, source_names, mimic_names, multipliers, offsets = (
            parse_mimic_joint(robot_urdf)
        )
        if self.ignore_mimic_joint:
            source_names, mimic_names, multipliers, offsets = [], [], [], []
        if self.joint_couplings:
            if not self.has_joint_limits:
                raise ValueError("joint_couplings requires has_joint_limits=true")
            joint_names, effective_joint_limits = configure_joint_couplings(
                robot, joint_names, effective_joint_limits, self.joint_couplings,
                source_names, mimic_names, multipliers, offsets,
            )

        # Geometry spans belong to angle terms, independently of qpos limits.
        angle_prior = None
        if self.joint_angle_weight > 0:
            from dex_retargeting.joint_angle_prior import BoneAnglePrior

            angle_prior = BoneAnglePrior(
                weight=self.joint_angle_weight,
                ranges=self.joint_angle_ranges,
                mapping=self.joint_angle_mapping,
            )

        if self.type == "position":
            optimizer = PositionOptimizer(
                robot,
                joint_names,
                target_link_names=self.target_link_names,
                target_link_human_indices=self.target_link_human_indices,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "vector":
            optimizer = VectorOptimizer(
                robot,
                joint_names,
                target_origin_link_names=self.target_origin_link_names,
                target_task_link_names=self.target_task_link_names,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
        elif self.type == "dexpilot":
            optimizer = DexPilotOptimizer(
                robot,
                joint_names,
                finger_tip_link_names=self.finger_tip_link_names,
                wrist_link_name=self.wrist_link_name,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                project_dist=self.project_dist,
                escape_dist=self.escape_dist,
            )
        else:
            raise RuntimeError()

        if angle_prior is not None:
            optimizer.set_joint_angle_prior(angle_prior)

        # Expose the hardware model to output adapters, without depending on
        # optimizer-internal joint ordering or the YAML target list.
        optimizer.joint_couplings = self.joint_couplings or {}

        if 0 <= self.low_pass_alpha <= 1:
            lp_filter = LPFilter(self.low_pass_alpha)
        else:
            lp_filter = None

        # Both FK and Jacobians include the follower's motion via the source.
        if mimic_names:
            adaptor = MimicJointKinematicAdaptor(
                robot,
                target_joint_names=joint_names,
                source_joint_names=source_names,
                mimic_joint_names=mimic_names,
                multipliers=multipliers,
                offsets=offsets,
            )
            optimizer.set_kinematic_adaptor(adaptor)

        retargeting = SeqRetargeting(
            optimizer,
            has_joint_limits=self.has_joint_limits,
            lp_filter=lp_filter,
            joint_limits_override=effective_joint_limits,
            strict_joint_limits=bool(self.joint_couplings),
        )
        return retargeting


def get_retargeting_config(config_path: Union[str, Path]) -> RetargetingConfig:
    config = RetargetingConfig.load_from_file(config_path)
    return config


def configure_joint_couplings(
    robot, joint_names, joint_limits, couplings,
    source_names, mimic_names, multipliers, offsets,
):
    """Validate couplings and intersect source limits with follower limits.

    Only direct couplings from independently optimized sources are supported.
    The four lists also contain active URDF mimic relations and are extended
    here so one kinematic adaptor handles both kinds of linkage.
    """
    if not isinstance(couplings, dict):
        raise ValueError("joint_couplings must be a mapping keyed by follower joint")
    robot_names = robot.dof_joint_names
    target_names = list(joint_names)
    for follower, spec in couplings.items():
        if not isinstance(spec, dict) or not {"source", "multiplier"} <= set(spec):
            raise ValueError(f"Coupling {follower} requires source and multiplier")
        if set(spec) - {"source", "multiplier", "offset"}:
            raise ValueError(f"Unknown coupling fields for {follower}")
        source = spec["source"]
        if follower not in robot_names or source not in robot_names:
            raise ValueError(f"Unknown coupling joint: {source} -> {follower}")
        if follower in mimic_names:
            raise ValueError(f"Coupling follower {follower} already has a URDF mimic")
        if follower not in target_names or source not in target_names:
            raise ValueError(f"Coupling source/follower must be target joints: {source} -> {follower}")
        multiplier = float(spec["multiplier"])
        offset = float(spec.get("offset", 0.0))
        if not np.isfinite([multiplier, offset]).all():
            raise ValueError(f"Coupling coefficients must be finite: {follower}")
        source_names.append(source)
        mimic_names.append(follower)
        multipliers.append(multiplier)
        offsets.append(offset)

    if set(source_names) & set(mimic_names):
        raise ValueError("Coupling chains/cycles are unsupported; sources must be independent")
    limits = joint_limits.copy()
    for source, follower, multiplier, offset in zip(
        source_names, mimic_names, multipliers, offsets
    ):
        if source not in target_names:
            raise ValueError(f"Coupling source must be optimized: {source}")
        source_id = robot_names.index(source)
        follower_id = robot_names.index(follower)
        follower_lower, follower_upper = limits[follower_id]
        if multiplier == 0:
            if not follower_lower <= offset <= follower_upper:
                raise ValueError(f"Infeasible constant coupling for {follower}")
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            allowed = np.sort((limits[follower_id] - offset) / multiplier)
        if not np.isfinite(allowed).all():
            raise ValueError(f"Coupling limit transformation is not finite: {follower}")
        lower = max(limits[source_id, 0], allowed[0])
        upper = min(limits[source_id, 1], allowed[1])
        if lower > upper:
            raise ValueError(f"Infeasible joint limits for coupling {source} -> {follower}")
        limits[source_id] = (lower, upper)
    target_names = [name for name in target_names if name not in couplings]
    return target_names, limits


def parse_mimic_joint(
    robot_urdf: urdf.URDF,
) -> Tuple[bool, List[str], List[str], List[float], List[float]]:
    mimic_joint_names = []
    source_joint_names = []
    multipliers = []
    offsets = []
    for name, joint in robot_urdf.joint_map.items():
        if joint.mimic is not None:
            mimic_joint_names.append(name)
            source_joint_names.append(joint.mimic.joint)
            multipliers.append(joint.mimic.multiplier)
            offsets.append(joint.mimic.offset)

    return (
        len(mimic_joint_names) > 0,
        source_joint_names,
        mimic_joint_names,
        multipliers,
        offsets,
    )

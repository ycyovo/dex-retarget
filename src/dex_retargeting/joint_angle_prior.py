"""Match human and robot bone angles independently of joint-axis conventions."""

from numbers import Integral
from typing import Any, Dict, Optional

import numpy as np
import torch


class BoneAnglePrior:
    """Unsigned successive-bone angles: straight = 0, folded back = pi.

    Entries describe pairs of geometric angles, not independent robot DOFs.
    All point triples run from the wrist toward the fingertip. Robot points
    are URDF link-frame origins evaluated by forward kinematics.
    """

    def __init__(
        self,
        weight: float,
        mapping: Dict[str, Dict[str, Any]],
        ranges: Optional[Dict[str, float]] = None,
    ):
        self.weight = float(weight)
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("BoneAnglePrior weight must be finite and positive")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Enabled bone angle prior requires joint_angle_mapping")
        self.angle_names = list(mapping)
        human_indices, robot_triples = [], []
        for name, spec in mapping.items():
            if not isinstance(spec, dict) or set(spec) != {"human_indices", "robot_links"}:
                raise ValueError(
                    f"Angle {name} requires human_indices and robot_links; "
                    "old type/indices/scale/offset joint mappings are unsupported"
                )
            indices = spec["human_indices"]
            if (
                not isinstance(indices, (list, tuple))
                or len(indices) != 3
                or any(
                    not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < 21
                    for i in indices
                )
                or len(set(indices)) != 3
            ):
                raise ValueError(
                    f"{name} requires three distinct human indices in [0, 20]"
                )
            links = spec["robot_links"]
            if (
                not isinstance(links, (list, tuple))
                or len(links) != 3
                or any(not isinstance(link, str) or not link for link in links)
                or len(set(links)) != 3
            ):
                raise ValueError(f"{name} requires three distinct robot link names")
            human_indices.append(indices)
            robot_triples.append(links)
        self.human_indices = np.asarray(human_indices, dtype=int)
        self.robot_triples = robot_triples
        self.robot_link_names = list(dict.fromkeys(
            link for triple in robot_triples for link in triple
        ))
        self.robot_indices = None

        ranges = {} if ranges is None else ranges
        unknown = set(ranges) - set(self.angle_names)
        if unknown:
            raise ValueError(
                f"joint_angle_ranges contains unknown angles: {sorted(unknown)}"
            )
        # Geometric unsigned angles have a natural domain [0, pi]. Joint qpos
        # limits (including locked [0, 0] joints) do not define these spans.
        self.ranges = np.array([
            np.pi if ranges.get(name) is None else float(ranges[name])
            for name in self.angle_names
        ])
        invalid = [
            name for name, span in zip(self.angle_names, self.ranges)
            if not np.isfinite(span) or span <= 1e-6 or span > np.pi
        ]
        if invalid:
            raise ValueError(
                f"joint_angle_ranges must be in (1e-6, pi] radians: {invalid}"
            )

    def bind_robot_links(self, computed_link_names):
        """Use the optimizer's shared FK positions and translational Jacobians."""
        self.robot_indices = torch.tensor([
            [computed_link_names.index(link) for link in triple]
            for triple in self.robot_triples
        ], dtype=torch.long)

    @staticmethod
    def _validate_bones(triples):
        incoming = triples[:, 1] - triples[:, 0]
        outgoing = triples[:, 2] - triples[:, 1]
        if np.any(np.linalg.norm(incoming, axis=1) < 1e-9) or np.any(
            np.linalg.norm(outgoing, axis=1) < 1e-9
        ):
            raise ValueError(
                "Cannot compute bone angle from coincident bone endpoints"
            )
        return incoming, outgoing

    @classmethod
    def angles_from_points(cls, triples):
        """Return angles for an (N, 3, 3) array of ordered point triples."""
        triples = np.asarray(triples, dtype=np.float64)
        if (
            triples.ndim != 3 or triples.shape[1:] != (3, 3)
            or not np.isfinite(triples).all()
        ):
            raise ValueError(
                "Bone point triples must be finite with shape (N, 3, 3)"
            )
        incoming, outgoing = cls._validate_bones(triples)
        return np.arctan2(
            np.linalg.norm(np.cross(incoming, outgoing), axis=1),
            np.sum(incoming * outgoing, axis=1),
        )

    def estimate_reference(self, human_joint_pos):
        points = np.asarray(human_joint_pos, dtype=np.float64)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError(
                "human_joint_pos must be a finite array of shape (21, 3)"
            )
        return self.angles_from_points(points[self.human_indices])

    def prepare_reference(self, reference_angles):
        if reference_angles is None:
            raise ValueError(
                "Enabled bone angle prior requires human_joint_pos or reference_angles"
            )
        reference = np.asarray(reference_angles, dtype=np.float64)
        if (
            reference.shape != (len(self.angle_names),)
            or not np.isfinite(reference).all()
            or np.any(reference < 0)
            or np.any(reference > np.pi)
        ):
            raise ValueError(
                f"reference_angles must contain {len(self.angle_names)} geometric "
                "angles in [0, pi], ordered by joint_angle_mapping entries"
            )
        return reference.copy()

    def robot_angles(self, body_pos):
        """Differentiable angles; URDF signs and fixed rotations enter via FK."""
        if self.robot_indices is None:
            raise RuntimeError("Bind robot links before computing bone angles")
        triples = body_pos[self.robot_indices]
        self._validate_bones(triples.detach().cpu().numpy())
        incoming = triples[:, 1] - triples[:, 0]
        outgoing = triples[:, 2] - triples[:, 1]
        # atan2 avoids the unbounded derivative of acos at straight/folded
        # bones. At exact collinearity norm uses a finite zero subgradient.
        return torch.atan2(
            torch.linalg.vector_norm(
                torch.cross(incoming, outgoing, dim=1), dim=1
            ),
            (incoming * outgoing).sum(dim=1),
        )

    def loss(self, body_pos, reference_angles):
        reference = torch.as_tensor(
            reference_angles, dtype=body_pos.dtype, device=body_pos.device
        )
        spans = torch.as_tensor(
            self.ranges, dtype=body_pos.dtype, device=body_pos.device
        )
        return self.weight * (
            ((self.robot_angles(body_pos) - reference) / spans)**2
        ).mean()

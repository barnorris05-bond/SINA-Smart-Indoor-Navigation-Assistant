"""
calibration/plan.py

EXPERIMENT TARGET-DISTANCE PLAN (Phase 10 preparation).

A deterministic schedule of target distances and repetitions for a
future controlled OAK-D measurement session: place a target at a known
physical distance, capture N observations, repeat for each distance.

Rules (milestone §6):

- Targets are EXPERIMENT TARGETS for measurement planning only. They
  are NOT navigation/risk/safety thresholds. This module imports
  NOTHING from navigation or risk configuration, and duplicates none of
  their values. (Enforced by a source-level test.)
- Values are declared by the operator per session; DEFAULTS are a
  starting point, explicitly labeled development/test, never validated
  or optimal.
- The plan is deterministic and inspectable; it does not generate
  measurements itself.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------
# DEFAULTS - DEVELOPMENT/TEST EXPERIMENT TARGETS ONLY.
# Not navigation thresholds. Not safety-calibrated. Simply a sensible
# indoor bench spread spanning SINA's near-field operating range.
# --------------------------------------------------------------
DEFAULT_EXPERIMENT_TARGETS_M: Tuple[float, ...] = (
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
)
DEFAULT_REPETITIONS = 10


@dataclass(frozen=True)
class TargetDistancePlan:
    """
    One experiment's distance schedule.

    targets_m:   nominal physical distances to place the target at.
    repetitions: observations captured per distance target.
    """

    targets_m: Tuple[float, ...] = DEFAULT_EXPERIMENT_TARGETS_M
    repetitions: int = DEFAULT_REPETITIONS

    def __post_init__(self) -> None:
        if not self.targets_m:
            raise ValueError("targets_m must contain at least one distance")
        if any(t <= 0 for t in self.targets_m):
            raise ValueError(f"targets must be positive, got {self.targets_m!r}")
        if len(set(self.targets_m)) != len(self.targets_m):
            raise ValueError(f"targets must be unique, got {self.targets_m!r}")
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")

    # ------------------------------------------------------
    # Inspection
    # ------------------------------------------------------

    def total_observations(self) -> int:
        """Maximum planned observations (targets x repetitions)."""
        return len(self.targets_m) * self.repetitions

    def summary(self) -> Dict[str, object]:
        """Deterministic description for reports."""
        return {
            "kind": "EXPERIMENT TARGETS - not navigation/safety thresholds",
            "targets_m": list(self.targets_m),
            "repetitions_per_target": self.repetitions,
            "total_planned_observations": self.total_observations(),
            "calibrated": False,
            "note": (
                "DEVELOPMENT/TEST experiment schedule; actual calibration "
                "requires REAL MEASURED OAK-D observations (Phase 9 "
                "hardware checkpoint currently BLOCKED)."
            ),
        }

    def checklist(self) -> List[Tuple[float, int]]:
        """
        Ordered (target_m, remaining_repetitions) capture checklist:
        every planned slot, for the operator to tick off in sequence.
        """
        return [(t, self.repetitions) for t in sorted(self.targets_m)]

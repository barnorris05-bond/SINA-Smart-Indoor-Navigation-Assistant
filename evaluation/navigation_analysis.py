"""
evaluation/navigation_analysis.py

DESCRIPTIVE navigation / risk / temporal statistics from EventLogger
traces (evaluation/event_log.py format: one JSON dict per frame with
"action" and "objects").

This is DESCRIPTIVE analysis only. It quantifies behavior; it never
claims navigation effectiveness, safety, or that temporal stabilization
improves anything - such claims require controlled comparisons that
this module does not make. compare_datasets() reports a "descriptive
comparison" between two labeled datasets and manufactures no causality.
"""

from typing import Dict, List, Optional, Sequence

# Display order for action names (project vocabulary: NavigationAction).
ACTION_ORDER = ("CONTINUE", "SLOW_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STOP")
RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def _action_sequence(events: Sequence[dict]) -> List[str]:
    return [str(e["action"]) for e in events if "action" in e]


def _runs(seq: Sequence[str]) -> List[Dict[str, object]]:
    """Maximal constant runs: [{value, length}] in order."""
    runs: List[Dict[str, object]] = []
    for value in seq:
        if runs and runs[-1]["value"] == value:
            runs[-1]["length"] += 1            # type: ignore[index]
        else:
            runs.append({"value": value, "length": 1})
    return runs


def _action_transitions(actions: Sequence[str]) -> Dict[str, int]:
    """Count of each A->B transition (sorted-key dict; excludes A->A)."""
    counts: Dict[str, int] = {}
    for a, b in zip(actions, actions[1:]):
        if a != b:
            key = f"{a}->{b}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ------------------------------------------------------------------
# Navigation statistics
# ------------------------------------------------------------------

def navigation_stats(events: Sequence[dict]) -> dict:
    """Descriptive action statistics (§13). No effectiveness claims."""
    actions = _action_sequence(events)
    if not actions:
        return {"frames": 0}

    counts = {a: 0 for a in ACTION_ORDER}
    for a in actions:
        counts.setdefault(a, 0)
        counts[a] += 1

    runs = _runs(actions)
    stop_runs = [r for r in runs if r["value"] == "STOP"]
    stop_frames = sum(int(r["length"]) for r in stop_runs)  # type: ignore[arg-type]

    # Number of A->B action changes == transitions between runs.
    # (len(actions) - len(runs) would wrongly count repeated frames.)
    transitions = _action_transitions(actions)
    action_change_count = sum(transitions.values())

    return {
        "frames": len(actions),
        "action_counts": counts,
        "action_transitions": transitions,
        "action_change_count": action_change_count,
        "longest_stable_run_frames": max(int(r["length"]) for r in runs),
        "stop_frames": stop_frames,
        "stop_run_count": len(stop_runs),
        "longest_stop_run_frames": (
            max(int(r["length"]) for r in stop_runs) if stop_runs else 0),
        "escalation_events": _escalation_count(actions),
    }


def _escalation_count(actions: Sequence[str]) -> int:
    """
    Transitions toward a higher-severity action (severity ladder from
    navigation/temporal_navigator.py: STOP 4, SLOW_DOWN/MOVE_* 2,
    CONTINUE 1). Descriptive counting only.
    """
    severity = {"STOP": 4, "SLOW_DOWN": 2, "MOVE_LEFT": 2,
                "MOVE_RIGHT": 2, "CONTINUE": 1}
    return sum(
        1 for a, b in zip(actions, actions[1:])
        if severity.get(b, 0) > severity.get(a, 0)
    )


# ------------------------------------------------------------------
# Risk statistics
# ------------------------------------------------------------------

def risk_stats(events: Sequence[dict]) -> dict:
    """
    Descriptive risk statistics over per-object annotations (§14).
    TTC stats only over present, positive values. No safety claims.
    """
    risk_counts = {r: 0 for r in RISK_ORDER}
    transitions: Dict[str, int] = {}
    ttcs: List[float] = []

    for event in events:
        for obj in event.get("objects", []):
            level = obj.get("risk_level")
            if level is not None:
                risk_counts.setdefault(level, 0)
                risk_counts[level] += 1
            ttc = obj.get("risk_ttc_s")
            if ttc is not None and ttc > 0:
                ttcs.append(float(ttc))

    # Risk transitions are tracked per object across frames using the
    # objects' identity (track_id when present, else label).
    previous: Dict[str, str] = {}
    for event in events:
        for obj in event.get("objects", []):
            level = obj.get("risk_level")
            if level is None:
                continue
            key = str(obj.get("track_id")
                      if obj.get("track_id") is not None
                      else obj.get("label"))
            prev = previous.get(key)
            if prev is not None and prev != level:
                tkey = f"{prev}->{level}"
                transitions[tkey] = transitions.get(tkey, 0) + 1
            previous[key] = level

    return {
        "risk_level_counts": dict(sorted(risk_counts.items())),
        "risk_transitions": dict(sorted(transitions.items())),
        "ttc_observations": len(ttcs),
        "ttc_min_s": min(ttcs) if ttcs else None,
        "ttc_median_s": (
            sorted(ttcs)[len(ttcs) // 2] if ttcs else None),
    }


# ------------------------------------------------------------------
# Temporal stabilization statistics
# ------------------------------------------------------------------

def temporal_stats(events: Sequence[dict]) -> dict:
    """
    Quantify temporal behavior (§15): change rate, stability, direction
    flips. Descriptive only - no claim that stabilization improves
    safety (that needs a controlled comparison; see compare_datasets).
    """
    actions = _action_sequence(events)
    if not actions:
        return {"frames": 0}

    lateral = {"MOVE_LEFT", "MOVE_RIGHT"}
    runs = _runs(actions)
    changes = max(0, len(runs) - 1)
    flips = sum(
        1 for a, b in zip(actions, actions[1:])
        if a in lateral and b in lateral and a != b
    )
    return {
        "frames": len(actions),
        "action_changes_per_frame": changes / len(actions),
        "longest_stable_action_run": max(int(r["length"]) for r in runs),
        "stop_run_count": sum(1 for r in runs if r["value"] == "STOP"),
        "direction_changes": flips,
    }


# ------------------------------------------------------------------
# Before/after comparison (descriptive)
# ------------------------------------------------------------------

def compare_datasets(
    baseline: Sequence[dict],
    treatment: Sequence[dict],
    baseline_name: str = "baseline",
    treatment_name: str = "treatment",
) -> dict:
    """
    Side-by-side descriptive comparison of two event traces (§16).

    The result is explicitly labeled a DESCRIPTIVE COMPARISON: equal
    scene scripts are the caller's responsibility and no causal claim
    ("improvement", "proven") is produced by this module.
    """
    a, b = navigation_stats(baseline), navigation_stats(treatment)
    at, bt = temporal_stats(baseline), temporal_stats(treatment)
    return {
        "comparison_type": "DESCRIPTIVE COMPARISON - no causal claim",
        "datasets": {
            baseline_name: {
                "navigation": a,
                "temporal": at,
            },
            treatment_name: {
                "navigation": b,
                "temporal": bt,
            },
        },
        "differences": {
            "action_changes": (
                None if not a or not b
                else (b["action_change_count"] - a["action_change_count"])),
            "direction_changes": (
                None if at.get("frames", 0) == 0 or bt.get("frames", 0) == 0
                else (bt["direction_changes"] - at["direction_changes"])),
            "stop_frames": (
                None if not a or not b
                else (b["stop_frames"] - a["stop_frames"])),
            "longest_stable_run": (
                None if not a or not b
                else (b["longest_stable_run_frames"]
                      - a["longest_stable_run_frames"])),
        },
    }

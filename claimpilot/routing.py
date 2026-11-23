"""Routing engine.

Takes triage scores and sends the claim somewhere. Rules are declared in
``config/routing_rules.yaml`` and evaluated in order; the first full match
wins.

The hard product rule is enforced structurally rather than by convention:

* the rules file declares ``allowed_destinations`` and ``forbidden_destinations``;
* :func:`load_rules` refuses to load a file containing any forbidden
  destination, or any destination not on the allow-list;
* :data:`Destination` has no denial member, so a denial cannot even be
  represented as a routing outcome.

Auto-approval is permitted. Auto-denial is not representable.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
import yaml

from claimpilot import config


class Destination(str, Enum):
    """Every outcome routing can produce. There is deliberately no denial."""

    STP_AUTO_APPROVE = "stp_auto_approve"
    ADJUSTER_QUEUE = "adjuster_queue"
    SENIOR_ADJUSTER = "senior_adjuster"
    SIU_QUEUE = "siu_queue"

    @property
    def label(self) -> str:
        return {
            "stp_auto_approve": "STP — auto-approved",
            "adjuster_queue": "Adjuster queue",
            "senior_adjuster": "Senior adjuster",
            "siu_queue": "SIU queue",
        }[self.value]

    @property
    def is_automated(self) -> bool:
        return self is Destination.STP_AUTO_APPROVE

    @property
    def requires_human(self) -> bool:
        return not self.is_automated


class AutoDenialError(RuntimeError):
    """Raised if a denial outcome is ever introduced into the rules."""


@dataclass
class RoutingDecision:
    claim_id: str
    destination: Destination
    rule_name: str
    reason: str
    rules_version: str
    matched_conditions: dict[str, Any] = field(default_factory=dict)

    @property
    def is_auto_approved(self) -> bool:
        return self.destination.is_automated

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "destination": self.destination.value,
            "destination_label": self.destination.label,
            "routing_rule": self.rule_name,
            "routing_reason": " ".join(self.reason.split()),
            "requires_human": self.destination.requires_human,
            "auto_approved": self.is_auto_approved,
            "rules_version": self.rules_version,
        }


@functools.lru_cache(maxsize=1)
def load_rules() -> dict[str, Any]:
    """Load and validate the routing rules.

    Refuses any file that could produce an automatic denial.
    """
    path = config.ROUTING_RULES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing routing rules at {path}")

    with path.open("r", encoding="utf-8") as fh:
        rules = yaml.safe_load(fh)

    allowed = {d.lower() for d in rules.get("allowed_destinations", [])}
    forbidden = {d.lower() for d in rules.get("forbidden_destinations", [])}

    if not allowed:
        raise ValueError("routing_rules.yaml must declare allowed_destinations")

    # The allow-list itself may not contain a denial.
    if overlap := allowed & forbidden:
        raise AutoDenialError(
            f"Denial destinations present in allowed_destinations: {sorted(overlap)}"
        )
    for name in allowed:
        if not _destination_is_valid(name):
            raise AutoDenialError(
                f"allowed_destinations contains '{name}', which is not a member of "
                "Destination. A denial outcome cannot be represented."
            )

    destinations: list[str] = [
        str(rule["destination"]).lower() for rule in rules.get("rules", [])
    ]
    destinations.append(str(rules["fallback"]["destination"]).lower())

    for dest in destinations:
        if dest in forbidden:
            raise AutoDenialError(
                f"Routing rule produces a forbidden denial destination: '{dest}'. "
                "The system can auto-approve but must never auto-deny."
            )
        if dest not in allowed:
            raise AutoDenialError(
                f"Routing rule produces undeclared destination '{dest}'."
            )

    return rules


def _destination_is_valid(name: str) -> bool:
    try:
        Destination(name)
    except ValueError:
        return False
    return True


def _matches(rule: dict[str, Any], triage: dict[str, Any]) -> bool:
    for key, accepted in rule.get("when", {}).items():
        value = triage.get(key)
        if value not in accepted:
            return False
    return True


def route(triage: dict[str, Any]) -> RoutingDecision:
    """Route one triaged claim. First matching rule wins."""
    rules = load_rules()

    for rule in rules["rules"]:
        if _matches(rule, triage):
            return RoutingDecision(
                claim_id=str(triage.get("claim_id", "")),
                destination=Destination(str(rule["destination"]).lower()),
                rule_name=str(rule["name"]),
                reason=str(rule["reason"]),
                rules_version=str(rules["version"]),
                matched_conditions=dict(rule.get("when", {})),
            )

    fallback = rules["fallback"]
    return RoutingDecision(
        claim_id=str(triage.get("claim_id", "")),
        destination=Destination(str(fallback["destination"]).lower()),
        rule_name="fallback",
        reason=str(fallback["reason"]),
        rules_version=str(rules["version"]),
    )


def route_batch(triage: pd.DataFrame) -> pd.DataFrame:
    """Route every triaged claim."""
    rows = [route(row).to_dict() for _, row in triage.iterrows()]
    return pd.DataFrame(rows)


def assert_no_auto_denial_path() -> None:
    """Explicit guarantee, asserted in the test suite and callable at runtime."""
    rules = load_rules()
    forbidden = {d.lower() for d in rules.get("forbidden_destinations", [])}

    reachable = {str(r["destination"]).lower() for r in rules["rules"]}
    reachable.add(str(rules["fallback"]["destination"]).lower())

    if bad := reachable & forbidden:
        raise AutoDenialError(f"Reachable denial destinations: {sorted(bad)}")

    for dest in reachable:
        if Destination(dest) not in set(Destination):
            raise AutoDenialError(f"Unknown destination reachable: {dest}")

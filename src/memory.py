"""Versioned, source-bound pattern memory. Promotions require external reviews."""

from __future__ import annotations

from pathlib import Path

import yaml


STATES = ("NEW", "INVESTIGATING", "PENDING_VERIFICATION", "VERIFIED", "REVOKED")
DECISIONS = ("MALICIOUS", "BENIGN")


class PatternMemory:
    def __init__(self, source: str, config: dict, entries: dict | None = None):
        self.source = source
        self.config = config
        self.entries = entries if entries is not None else {}
        self.revocations = 0
        self.verification_cost_usd = 0.0

    def provisional(self, key: str, flow_id: str, verdict: str, references: list[str],
                    window: int, attack_family_candidate: str | None = None,
                    attack_family_evidence_refs: list[str] | None = None,
                    family_failure_reason: str | None = None) -> None:
        if verdict not in DECISIONS:
            return
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= int(self.config["memory_max_entries"]):
                return
            entry = {"state": "PENDING_VERIFICATION", "scope": self.source,
                     "created_window": window, "updated_window": window,
                     "provisional": [], "reviews": [], "verdict": None}
            self.entries[key] = entry
        if entry["state"] in ("VERIFIED", "REVOKED"):
            return
        record = {"flow_id": flow_id, "verdict": verdict,
                  "evidence_refs": references, "window": window}
        if attack_family_candidate is not None:
            record["attack_family_candidate"] = attack_family_candidate
            record["attack_family_evidence_refs"] = attack_family_evidence_refs or []
        if family_failure_reason is not None:
            record["family_failure_reason"] = family_failure_reason
        entry["provisional"].append(record)
        entry["state"] = "PENDING_VERIFICATION"
        entry["updated_window"] = window

    def review(self, record: dict, window: int,
               allowed_source_ids: dict[str, str]) -> bool:
        """Review records are external assertions, never dataset evaluation labels."""
        required = {"pattern", "sample_id", "group_id", "reviewer_id", "verdict",
                    "source_partition", "independent_review", "available_window"}
        if set(record) != required or record["independent_review"] is not True:
            raise ValueError("Review lacks independent-verification fields")
        if (record["source_partition"] not in ("training", "calibration") or
                allowed_source_ids.get(record["sample_id"]) != record["source_partition"] or
                record["verdict"] not in DECISIONS or
                int(record["available_window"]) > window):
            raise ValueError("Review overlaps evaluation or is unavailable")
        key = record["pattern"]
        entry = self.entries.get(key)
        if entry is None or entry["scope"] != self.source:
            raise ValueError("Review pattern is outside source memory scope")
        if any(item["flow_id"] == record["sample_id"] for item in entry["provisional"]):
            raise ValueError("Review cannot reuse the investigated flow")
        identity = (record["sample_id"], record["reviewer_id"])
        if (entry["state"] == "REVOKED" or
                window < entry["created_window"] + int(self.config["verification_delay_windows"]) or
                any((r["sample_id"], r["reviewer_id"]) == identity for r in entry["reviews"]) or
                self.verification_cost_usd + float(self.config["verification_cost_usd"]) >
                float(self.config["max_verification_cost_usd"])):
            return False
        entry["reviews"].append(record.copy())
        entry["updated_window"] = window
        self.verification_cost_usd += float(self.config["verification_cost_usd"])
        provisional = {item["verdict"] for item in entry["provisional"]}
        if provisional and (len(provisional) > 1 or record["verdict"] not in provisional):
            self.revoke(key, window, "independent_review_disagrees_with_agent")
            return False
        verdicts = {review["verdict"] for review in entry["reviews"]}
        if len(verdicts) > 1:
            self.revoke(key, window, "conflicting_independent_reviews")
            return False
        if entry["state"] == "VERIFIED" and entry["verdict"] not in verdicts:
            self.revoke(key, window, "contradictory_review")
            return False
        groups = {review["group_id"] for review in entry["reviews"]}
        reviewers = {review["reviewer_id"] for review in entry["reviews"]}
        samples = {review["sample_id"] for review in entry["reviews"]}
        needed = int(self.config["min_independent_verifications"])
        if entry["state"] == "PENDING_VERIFICATION" and min(len(groups), len(reviewers), len(samples)) >= needed:
            entry["state"] = "VERIFIED"
            entry["verdict"] = next(iter(verdicts))
            entry["verified_window"] = window
            return True
        return False

    def resolve(self, key: str, window: int) -> str | None:
        entry = self.entries.get(key)
        if entry is None or entry["state"] != "VERIFIED":
            return None
        if window - entry["verified_window"] >= int(self.config["memory_ttl_windows"]):
            self.revoke(key, window, "expired")
            return None
        return entry["verdict"]

    def revoke(self, key: str, window: int, reason: str) -> None:
        entry = self.entries[key]
        if entry["state"] != "REVOKED":
            entry["state"] = "REVOKED"
            entry["verdict"] = None
            entry["updated_window"] = window
            entry["revocation_reason"] = reason
            self.revocations += 1

    def document(self, model_sha256: str, policy_sha256: str, reference_sha256: str) -> dict:
        return {"version": "zeta-progressive-memory-v1", "source": self.source,
                "model_sha256": model_sha256, "policy_sha256": policy_sha256,
                "reference_sha256": reference_sha256, "entries": self.entries,
                "revocation_events": self.revocations,
                "verification_cost_usd": round(self.verification_cost_usd, 8)}

    @classmethod
    def load(cls, path: str, source: str, config: dict, hashes: tuple[str, str, str]):
        document = yaml.safe_load(Path(path).read_text())
        if (not isinstance(document, dict) or
                document.get("version") != "zeta-progressive-memory-v1" or
                document.get("source") != source or
                tuple(document.get(name) for name in
                      ("model_sha256", "policy_sha256", "reference_sha256")) != hashes):
            raise ValueError("Memory source or frozen artifact binding differs")
        entries = document.get("entries")
        if not isinstance(entries, dict) or len(entries) > int(config["memory_max_entries"]):
            raise ValueError("Memory entry table is invalid")
        needed = int(config["min_independent_verifications"])
        for key, entry in entries.items():
            if (not isinstance(key, str) or not key.startswith(source + ":") or
                    not isinstance(entry, dict) or entry.get("scope") != source or
                    entry.get("state") not in STATES or
                    not isinstance(entry.get("provisional"), list) or
                    not isinstance(entry.get("reviews"), list)):
                raise ValueError("Memory entry has invalid source or state")
            if entry["state"] == "VERIFIED":
                verdict = entry.get("verdict")
                reviews = entry["reviews"]
                if (verdict not in DECISIONS or
                        any(not isinstance(item, dict) for item in entry["provisional"] + reviews) or
                        not any(item.get("verdict") == verdict for item in entry["provisional"]) or
                        any(item.get("verdict") != verdict or
                            item.get("source_partition") not in ("training", "calibration") or
                            item.get("independent_review") is not True for item in reviews) or
                        min(len({item.get("sample_id") for item in reviews}),
                            len({item.get("group_id") for item in reviews}),
                            len({item.get("reviewer_id") for item in reviews})) < needed or
                        not isinstance(entry.get("verified_window"), int)):
                    raise ValueError("Verified memory lacks independent corroboration")
        memory = cls(source, config, document["entries"])
        memory.revocations = int(document["revocation_events"])
        memory.verification_cost_usd = float(document["verification_cost_usd"])
        if memory.verification_cost_usd > float(config["max_verification_cost_usd"]):
            raise ValueError("Stored verification cost exceeds configured cap")
        return memory

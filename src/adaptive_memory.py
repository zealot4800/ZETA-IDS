"""Persistent provisional attack/benign patterns updated by the controller."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3

from .evidence import digest


class AdaptiveMemory:
    def __init__(self, path: Path, cfg: dict, source: dict, dataset: str, version: str):
        self.options = cfg["adaptive_memory"]
        self.source = source
        self.scope = digest([dataset, version, source, cfg])
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS active_patterns (
            id TEXT PRIMARY KEY, scope TEXT NOT NULL, signature TEXT NOT NULL,
            status TEXT NOT NULL, decision TEXT, observations INTEGER NOT NULL,
            agreements INTEGER NOT NULL, disagreements INTEGER NOT NULL,
            probability_sum REAL NOT NULL, selection TEXT, expires_at TEXT NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS pattern_observations (
            pattern_id TEXT NOT NULL, features_hash TEXT NOT NULL,
            PRIMARY KEY(pattern_id, features_hash))""")
        self.db.commit()

    def close(self):
        self.db.close()

    def signature(self, flow: dict) -> tuple[str, dict]:
        bands = {}
        for name in self.source["feature_names"]:
            value = flow["features"][name]
            quantiles = self.source["baseline"][name].get("quantiles")
            if type(value) in (int, float) and math.isfinite(value) and quantiles:
                bounds = sorted(v for v in quantiles.values()
                                if type(v) in (int, float) and math.isfinite(v))
                median = quantiles.get("0.5", 0.0)
                median = median if type(median) in (int, float) else 0.0
                scale = max(bounds[-1] - bounds[0], 1.0) if bounds else 1.0
                # One robust numeric band handles both central and tail values.
                bands[name] = round(math.asinh((value - median) / scale), 1)
            else:
                bands[name] = value
        signature = {"tree_leaf": flow["dataplane_evidence"]["tree_leaf"], "bands": bands}
        return digest([self.scope, signature]), signature

    def _row(self, pattern_id):
        row = self.db.execute("SELECT * FROM active_patterns WHERE id=?", (pattern_id,)).fetchone()
        if row and datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            with self.db:
                self.db.execute("DELETE FROM active_patterns WHERE id=?", (pattern_id,))
                self.db.execute("DELETE FROM pattern_observations WHERE pattern_id=?", (pattern_id,))
            return None
        return row

    def lookup(self, flow: dict) -> dict:
        pattern_id, _ = self.signature(flow)
        row = self._row(pattern_id)
        if row is None:
            return {"pattern_id": pattern_id, "matched": False, "audit": False,
                    "selection": None, "status": None}
        support = flow.get("controller_context") or {}
        compatible = support.get("supported", False) and support.get("decision") == row["decision"]
        active = row["status"] == "provisional" and compatible
        if row["status"] == "provisional" and not compatible:
            self.suspend(flow)
        # An explanatory report can be reused before binary-rule promotion.
        # This also covers model-supported LLM abstentions, which cannot teach
        # an authoritative memory decision by themselves.
        report_compatible = (support.get("supported", False) and
                             support.get("decision") == "ATTACK" and
                             row["decision"] in (None, "ATTACK") and
                             row["status"] != "demoted")
        fraction = int(digest([flow["flow_id"], pattern_id])[:16], 16) / 2**64
        return {"pattern_id": pattern_id, "matched": active,
                "audit": active and fraction < self.options["audit_sample_rate"],
                "decision": row["decision"],
                "status": row["status"] if compatible else "support_required",
                "p_attack": row["probability_sum"] / max(row["observations"], 1),
                "selection": (json.loads(row["selection"]) if row["selection"] and
                              report_compatible else None)}

    def suspend(self, flow: dict):
        """A failed audit or changed source evidence immediately stops reuse."""
        pattern_id, _ = self.signature(flow)
        with self.db:
            self.db.execute("UPDATE active_patterns SET status='demoted', selection=NULL WHERE id=?",
                            (pattern_id,))

    def observe(self, flow: dict, decision: str, p_attack: float | None,
                selection: dict | None) -> dict:
        pattern_id, signature = self.signature(flow)
        row = self._row(pattern_id)
        strong = (p_attack is not None and
                  ((decision == "ATTACK" and p_attack >= self.options["attack_min_probability"]) or
                   (decision == "BENIGN" and p_attack <= self.options["benign_max_probability"])))
        support = flow.get("controller_context") or {}
        corroborated = support.get("supported", False) and support.get("decision") == decision
        strong = strong and corroborated
        observations = row["observations"] if row else 0
        agreements = row["agreements"] if row else 0
        disagreements = row["disagreements"] if row else 0
        probability_sum = row["probability_sum"] if row else 0.0
        learned = row["decision"] if row else None
        if learned is None and strong:
            learned = decision
        features_hash = digest(flow["features"])
        previous = self.db.execute(
            "SELECT 1 FROM pattern_observations WHERE pattern_id=? AND features_hash=?",
            (pattern_id, features_hash)).fetchone()
        saved = json.loads(row["selection"]) if row and row["selection"] else None
        contradiction = learned is not None and (decision != learned or not corroborated)
        if contradiction:
            saved = None
        elif support.get("decision") != "ATTACK":
            saved = None
        elif selection is not None and decision in ("ATTACK", "UNCERTAIN"):
            saved = selection
        # Replaying an identical flow or cached answer cannot promote a pattern.
        if previous is None and p_attack is not None and learned is not None:
            observations += 1
            probability_sum += p_attack
            if strong and decision == learned:
                agreements += 1
            elif learned is not None:
                disagreements += 1
        agreement = agreements / max(observations, 1)
        required = self.options["min_observations"][learned or "ATTACK"]
        if (contradiction or (row and row["status"] == "demoted") or
                disagreements >= self.options["max_disagreements"]):
            status = "demoted"
        elif (not contradiction and learned is not None and
              agreements >= required and
              agreement >= self.options["min_agreement"]):
            status = "provisional"
        else:
            status = "observing"
        expires = (row["expires_at"] if row else
                   (datetime.now(timezone.utc) + timedelta(days=self.options["ttl_days"])).isoformat())
        with self.db:
            self.db.execute("""INSERT OR REPLACE INTO active_patterns VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                pattern_id, self.scope, json.dumps(signature, allow_nan=False), status, learned,
                observations, agreements, disagreements, probability_sum,
                json.dumps(saved, allow_nan=False) if saved else None, expires))
            if p_attack is not None and learned is not None:
                self.db.execute("INSERT OR IGNORE INTO pattern_observations VALUES (?, ?)",
                                (pattern_id, features_hash))
        return {"pattern_id": pattern_id, "status": status,
                "observations": observations, "agreement": agreement}

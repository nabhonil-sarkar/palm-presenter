"""Temporal measurements and presentation rules, independently testable without a camera."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import sqlite3
import numpy as np


def unit_vector(values: np.ndarray) -> np.ndarray:
    """Normalize one finite non-zero embedding; reject invalid profile data."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 128 or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Expected a finite, non-zero 128-value face embedding.")
    return vector / norm


class ProfileDatabase:
    """Session-local authorized profiles in SQLite; no photos or disk database.

    Enrollment is explicitly requested by the user. Five consistent samples form
    each template. Example: db.enroll('Presenter', five_embeddings).
    """

    def __init__(self, threshold: float = 0.45, margin: float = 0.05):
        self.threshold, self.margin = threshold, margin
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE profiles(name TEXT PRIMARY KEY, vector BLOB)"
        )

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> None:
        name = name.strip()
        if not 1 <= len(name) <= 40 or len(embeddings) < 5:
            raise ValueError("Use a 1-40 character name and at least five samples.")
        if len(self.names()) >= 10 and name not in self.names():
            raise ValueError("Maximum ten session profiles. Reset profiles first.")
        vectors = np.vstack([unit_vector(e) for e in embeddings])
        if np.min(vectors @ vectors[0]) < 0.55:
            raise ValueError("Enrollment changed too much; face the camera and retry.")
        template = unit_vector(vectors.mean(axis=0))
        self.connection.execute(
            "INSERT OR REPLACE INTO profiles VALUES (?,?)", (name, template.tobytes())
        )
        self.connection.commit()

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        vector = unit_vector(embedding)
        scores = sorted(
            [
                (float(vector @ np.frombuffer(data, dtype=np.float32)), name)
                for name, data in self.connection.execute(
                    "SELECT name,vector FROM profiles"
                )
            ],
            reverse=True,
        )
        if not scores:
            return None, 0.0
        score, name = scores[0]
        gap = score - scores[1][0] if len(scores) > 1 else 2.0
        return (name if score >= self.threshold and gap >= self.margin else None), score

    def names(self) -> list[str]:
        return [
            r[0]
            for r in self.connection.execute("SELECT name FROM profiles ORDER BY name")
        ]

    def clear(self) -> None:
        self.connection.execute("DELETE FROM profiles")
        self.connection.commit()


@dataclass
class BlinkMeter:
    """Hysteresis counts completed brief closures, with gaps excluded from exposure."""

    closed_since: float | None = None
    previous: float | None = None
    exposure: float = 0.0
    count: int = 0

    def update(self, ear: float | None, now: float) -> dict:
        if self.previous is not None and now <= self.previous:
            raise ValueError("Blink timestamps must increase.")
        gap = now - self.previous if self.previous is not None else 0.0
        self.previous = now
        if ear is None or not math.isfinite(ear) or gap > 0.5:
            self.closed_since = None
            return self.snapshot(now)
        self.exposure += gap
        if ear < 0.21 and self.closed_since is None:
            self.closed_since = now
        elif ear > 0.24 and self.closed_since is not None:
            duration = now - self.closed_since
            self.count += int(0.06 <= duration <= 0.8)
            self.closed_since = None
        return self.snapshot(now)

    def snapshot(self, now: float) -> dict:
        return {
            "blink_count": self.count,
            "blink_rate": round(60 * self.count / self.exposure, 1)
            if self.exposure >= 10
            else None,
            "observed_seconds": round(self.exposure, 1),
            "closed_seconds": round(now - self.closed_since, 2)
            if self.closed_since is not None
            else 0.0,
        }


@dataclass
class PresenterOrientation:
    """Use a wider exit band and a short grace period for noisy pose readings."""

    facing: bool = False
    last_good: float = float("-inf")

    def update(self, pitch, yaw, reliable, now):
        if reliable:
            pitch_limit, yaw_limit = (38, 45) if self.facing else (30, 35)
            if abs(pitch) <= pitch_limit and abs(yaw) <= yaw_limit:
                self.last_good = now
                self.facing = True
            elif abs(pitch) > 60 or abs(yaw) > 65:
                self.facing = False
        if now - self.last_good > 0.45:
            self.facing = False
        return self.facing


@dataclass
class HandHistory:
    samples: deque = field(default_factory=lambda: deque(maxlen=40))
    pinch: bool = False
    last_event: float = -100.0
    last_pinch: float = -100.0
    label: str = "None"
    stable_since: float = 0.0
    last_seen: float = -100.0


class GestureTracker:
    """Convert landmark trajectories into edge-triggered pinch/swipe events.

    Distances are normalized by palm width; swipe direction is in displayed
    (mirrored) screen coordinates. Separate histories prevent hand mixing.
    """

    def __init__(self):
        self.hands: dict[str, HandHistory] = {}

    @staticmethod
    def open_fingers(points):
        # Use the three long fingers when the static model is uncertain in motion.
        extended = 0
        for base in (5, 9, 13):
            upper = points[base + 1] - points[base]
            lower = points[base + 3] - points[base + 1]
            lengths = np.linalg.norm(upper) * np.linalg.norm(lower)
            if lengths > 1 and np.dot(upper, lower) > 0.55 * lengths:
                extended += 1
        return extended == 3

    @staticmethod
    def sideways_thumb(points, model_label):
        """Detect a thumb pointing across the mirrored image with curled fingers."""
        if model_label not in {"Thumb Up", "Other", "Uncertain", "Fist"}:
            return None
        norm = np.linalg.norm
        palm_size = max(norm(points[9] - points[0]), norm(points[5] - points[17]), 1)
        dx, dy = points[4] - points[2]
        if abs(dx) < 0.48 * palm_size or abs(dx) < 1.4 * abs(dy):
            return None
        if norm(points[4] - points[5]) < 0.52 * palm_size:
            return None
        folded = []
        for base in (5, 9, 13, 17):
            finger = points[base : base + 4]
            length = norm(np.diff(finger, axis=0), axis=1).sum()
            folded.append(length > 1 and norm(finger[-1] - finger[0]) < 0.78 * length)
        if folded[0] and sum(folded) >= 3:
            return "Thumb Right" if dx > 0 else "Thumb Left"
        return None

    @staticmethod
    def swipe(samples):
        # Consider recent starts so a pause before moving does not dilute a swipe.
        end_time, end_x, end_y, end_width = samples[-1]
        for index, (start_time, x, y, width) in enumerate(samples):
            if not 0.09 <= end_time - start_time <= 1.0:
                continue
            dx, dy = end_x - x, end_y - y
            minimum = max(36, 1.05 * (width + end_width) / 2)
            if abs(dx) < minimum or abs(dy) > 0.60 * abs(dx):
                continue
            segment = list(samples)[index:]
            distance = sum(abs(b[1] - a[1]) for a, b in zip(segment, segment[1:]))
            if abs(dx) >= 0.8 * distance:
                return "Swipe Right" if dx > 0 else "Swipe Left"
        return None

    def update(self, key: str, points: np.ndarray, label: str, now: float) -> dict:
        points = np.asarray(points, dtype=float)
        if points.shape != (21, 2) or not np.isfinite(points).all():
            raise ValueError("Expected 21 finite hand points in pixel coordinates.")
        state = self.hands.setdefault(key, HandHistory())
        if now <= state.last_seen:
            raise ValueError("Hand timestamps must increase.")
        if now - state.last_seen > 0.5:
            state.samples.clear()
            state.pinch = False
            state.label = "None"
        state.last_seen = now
        width = max(float(np.linalg.norm(points[5] - points[17])), 1.0)
        ratio = float(np.linalg.norm(points[4] - points[8]) / width)
        direction = self.sideways_thumb(points, label)
        if direction:
            label = direction
        event = None
        if direction:
            state.pinch = False
        elif not state.pinch and ratio < 0.28:
            state.pinch = True
            event = "Pinch"
        elif state.pinch and ratio > 0.42:
            state.pinch = False
        if label != state.label:
            state.label = label
            state.stable_since = now
        palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
        open_palm = label == "Open Palm" or (
            label in {"Other", "Uncertain"} and self.open_fingers(points)
        )
        if open_palm and not state.pinch:
            state.samples.append((now, float(palm[0]), float(palm[1]), width))
            while now - state.samples[0][0] > 1.0:
                state.samples.popleft()
            event = self.swipe(state.samples)
            if event:
                state.samples.clear()
        elif (
            label not in {"Other", "Uncertain"}
            or state.pinch
            or (state.samples and now - state.samples[-1][0] > 0.32)
        ):
            state.samples.clear()
        if event is not None:
            previous = state.last_pinch if event == "Pinch" else state.last_event
            if now - previous < 0.8:
                event = None
            elif event == "Pinch":
                state.last_pinch = now
            else:
                state.last_event = now
        return {
            "static": label,
            "event": event,
            "pinching": state.pinch,
            "pinch_ratio": round(ratio, 3),
            "stable_seconds": now - state.stable_since,
            "pointer": points[8].tolist(),
        }

    def prune(self, now: float) -> None:
        self.hands = {
            key: state
            for key, state in self.hands.items()
            if now - state.last_seen <= 1.0
        }


SLIDES = [
    (
        "A presentation without a remote",
        "Face verification enables control. Hand gestures drive the presentation.",
    ),
    (
        "See the input",
        "The camera requests 30 FPS. Capture rate and analysis rate are measured separately.",
    ),
    (
        "Recognize the presenter",
        "A local session profile uses five face samples. Unknown faces cannot issue commands.",
    ),
    (
        "Make actions deliberate",
        "Hold a pose briefly. Swipe to move slides. Pinch once to toggle the spotlight.",
    ),
    (
        "Measure what matters",
        "Test different lighting, distances and camera angles. Report failures as well as successes.",
    ),
]


class PresentationController:
    """Identity and orientation gate with dwell, cooldown and release-to-rearm."""

    def __init__(self):
        self.slide = 0
        self.blackout = False
        self.spotlight = False
        self.armed = False
        self.last_action = "Ready"
        self.identity: str | None = None
        self.identity_since = 0.0
        self.thumb_latched = False
        self.thumb_release_since = None
        self.last_command = -100.0
        self.latched: str | None = None
        self.log: deque = deque(maxlen=100)

    def update(
        self,
        identity: str | None,
        facing: bool,
        face_count: int,
        hands: list[dict],
        now: float,
        fresh: bool = True,
    ) -> dict:
        if identity != self.identity:
            self.identity = identity
            self.identity_since = now
            self.armed = False
        gate = bool(identity and facing and face_count == 1 and fresh)
        reason = (
            "Ready"
            if gate
            else (
                "No face"
                if face_count == 0
                else "Multiple faces"
                if face_count != 1
                else "Unknown face"
                if not identity
                else "Look towards camera"
                if not facing
                else "Tracking stale"
            )
        )
        static_pose = hands[0]["static"] if len(hands) == 1 else None
        sideways = static_pose in {"Thumb Left", "Thumb Right"}
        if sideways:
            self.thumb_release_since = None
        elif self.thumb_latched:
            if self.thumb_release_since is None:
                self.thumb_release_since = now
            elif now - self.thumb_release_since >= 0.3:
                self.thumb_latched = False
        if not gate:
            self.armed = False
            self.latched = None
            self.identity_since = now
        elif now - self.identity_since < 0.3:
            reason = "Verifying presenter"
        elif len(hands) != 1:
            reason = "Show one control hand" if hands else "Show a gesture"
            self.latched = None
        else:
            hand = hands[0]
            static = hand["static"]
            if static != self.latched:
                self.latched = None
            event = hand.get("event")
            ready = hand.get("stable_seconds", 0) >= 0.35
            action = None
            if event and self.armed:
                action = {
                    "Swipe Left": "Next slide",
                    "Swipe Right": "Previous slide",
                    "Pinch": "Spotlight",
                }.get(event)
            elif (
                sideways
                and self.armed
                and not self.thumb_latched
                and hand.get("stable_seconds", 0) >= 0.45
            ):
                action = "Next slide" if static == "Thumb Right" else "Previous slide"
            elif ready and self.latched is None:
                if static == "Thumb Up":
                    action = "Enable control"
                elif static == "Fist":
                    action = "Pause control"
                elif self.armed and static == "Peace":
                    action = "Blackout"
            first_swipe = self.last_action == "Enable control" and action in {
                "Next slide",
                "Previous slide",
            }
            if action and (first_swipe or now - self.last_command >= 0.8):
                if action == "Enable control":
                    self.armed = True
                elif action == "Pause control":
                    self.armed = False
                elif action == "Next slide":
                    self.slide = min(len(SLIDES) - 1, self.slide + 1)
                elif action == "Previous slide":
                    self.slide = max(0, self.slide - 1)
                elif action == "Blackout":
                    self.blackout = not self.blackout
                elif action == "Spotlight":
                    self.spotlight = not self.spotlight
                if sideways:
                    self.thumb_latched = True
                self.last_action = action
                self.last_command = now
                self.latched = static
                self.log.append(
                    {
                        "seconds": round(now, 3),
                        "action": action,
                        "slide": self.slide + 1,
                    }
                )
            reason = "Control enabled" if self.armed else "Thumb up to enable"
        return self.snapshot(reason)

    def snapshot(self, reason: str = "Ready") -> dict:
        title, body = SLIDES[self.slide]
        return {
            "slide": self.slide + 1,
            "total": len(SLIDES),
            "title": title,
            "body": body,
            "blackout": self.blackout,
            "spotlight": self.spotlight,
            "armed": self.armed,
            "status": reason,
            "last_action": self.last_action,
        }

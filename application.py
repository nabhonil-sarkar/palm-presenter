"""Bounded local processing and an authenticated loopback-only web interface."""

import asyncio
from collections import Counter, deque
import secrets
import socket
import threading
import time
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from logic import (
    ProfileDatabase,
    BlinkMeter,
    PresenterOrientation,
    GestureTracker,
    PresentationController,
)


class PresenterApplication:
    """Own all mutable state; one lock prevents simultaneous model calls."""

    def __init__(self, models):
        self.models = models
        self.lock = threading.Lock()
        self.profiles = ProfileDatabase()
        self.calibration = {}
        self.reset()

    def reset(self, forget: bool = True) -> None:
        self.controller = PresentationController()
        self.gestures = GestureTracker()
        self.orientation = PresenterOrientation()
        self.orientation_owner = None
        self.blinks = {}
        self.hand_positions = {}
        self.face_positions = {}
        self.next_id = 0
        self.enrollment = None
        self.trial = None
        self.trials = []
        self.frame_times = deque(maxlen=600)
        self.processing_ms = deque(maxlen=600)
        self.started = time.monotonic()
        self.last_sequence = -1
        self.last_summary = {}
        self.frames = 0
        self.last_frame = -100.0
        if forget:
            self.profiles.clear()
            self.calibration.clear()

    def request_enrollment(self, name: str, consent: bool) -> str:
        if not consent:
            raise ValueError("Confirm permission to enroll your face.")
        name = name.strip()
        if not 1 <= len(name) <= 40:
            raise ValueError("Enter a name with 1-40 characters.")
        self.enrollment = {
            "name": name,
            "samples": [],
            "poses": [],
            "ears": [],
            "last": -100.0,
        }
        self.controller.armed = False
        return "Look at the camera with eyes open. Collecting five samples."

    def track(
        self, position, store: dict, now: float, used: set, max_distance=100
    ) -> str:
        candidates = [
            (float(np.linalg.norm(position - old[0])), key)
            for key, old in store.items()
            if now - old[1] < 0.5 and key not in used
        ]
        distance, key = min(candidates) if candidates else (float("inf"), "")
        if distance > max_distance:
            self.next_id += 1
            key = str(self.next_id)
        store[key] = (position, now)
        used.add(key)
        return key

    def process(self, frame: np.ndarray, sequence: int) -> dict:
        start = time.monotonic()
        if sequence <= self.last_sequence:
            raise ValueError("Out-of-order frame rejected.")
        self.last_sequence = sequence
        if (
            frame.ndim != 3
            or frame.shape[2] != 3
            or not 64 <= frame.shape[0] <= 720
            or not 64 <= frame.shape[1] <= 1280
        ):
            raise ValueError("Use a 64-1280 by 64-720 color frame.")
        if start - self.last_frame > 0.5:
            self.controller.armed = False
            self.controller.identity_since = start
            self.orientation = PresenterOrientation()
        self.last_frame = start
        faces = self.models.analyze_faces(frame)
        hands = self.models.analyze_hands(frame, mirrored=True)
        if len(faces) != 1:
            self.orientation = PresenterOrientation()
            self.orientation_owner = None
        message = ""
        used = set()
        for face in faces:
            x, y, w, h = face["box"]
            key = self.track(
                np.array([x + w / 2, y + h / 2]), self.face_positions, start, used
            )
            face["track"] = key
            identity, score = self.profiles.match(face["embedding"])
            face["identity"] = identity
            face["similarity"] = round(score, 3)
            meter = self.blinks.setdefault(key, BlinkMeter())
            ear = face["ear"]
            baseline = self.calibration.get(identity)
            normalized_ear = (
                ear * 0.28 / baseline["ear"] if baseline and ear is not None else ear
            )
            face.update(meter.update(normalized_ear, start))
            pose = face["pose"]
            face["facing"] = False
            if len(faces) == 1 and identity != self.orientation_owner:
                self.orientation = PresenterOrientation()
                self.orientation_owner = identity
            if baseline and len(faces) == 1:
                reliable = (
                    pose["pitch"] is not None
                    and pose["yaw"] is not None
                    and pose["error"] is not None
                    and pose["error"] < max(12, frame.shape[1] * 0.04)
                )
                pitch = yaw = None
                if reliable:
                    pitch = (pose["pitch"] - baseline["pitch"] + 180) % 360 - 180
                    yaw = (pose["yaw"] - baseline["yaw"] + 180) % 360 - 180
                    face["relative_pitch"] = round(pitch, 1)
                    face["relative_yaw"] = round(yaw, 1)
                face["facing"] = self.orientation.update(pitch, yaw, reliable, start)
        if self.enrollment:
            e = self.enrollment
            message = f"Enrolling {e['name']}: {len(e['samples'])}/5 samples"
            if len(faces) != 1:
                message = "Enrollment needs exactly one visible face."
            elif faces[0]["ear"] is None or faces[0]["pose"]["pitch"] is None:
                message = "Eyes and face landmarks must be visible."
            elif start - e["last"] >= 0.35:
                face = faces[0]
                e["samples"].append(face["embedding"])
                e["poses"].append(face["pose"])
                e["ears"].append(face["ear"])
                e["last"] = start
                if len(e["samples"]) >= 5:
                    try:
                        self.profiles.enroll(e["name"], e["samples"])
                        self.calibration[e["name"]] = {
                            "pitch": float(np.median([p["pitch"] for p in e["poses"]])),
                            "yaw": float(np.median([p["yaw"] for p in e["poses"]])),
                            "ear": max(0.08, float(np.median(e["ears"]))),
                        }
                        message = f"Enrolled {e['name']}. Thumb up enables control."
                    except ValueError as exc:
                        message = str(exc)
                    self.enrollment = None
        used = set()
        hand_results = []
        for hand in hands:
            xy = hand["points"]
            palm = xy[[0, 5, 9, 13, 17]].mean(axis=0)
            reach = min(220, max(100, 2.5 * np.linalg.norm(xy[5] - xy[17])))
            key = self.track(palm, self.hand_positions, start, used, max_distance=reach)
            temporal = self.gestures.update(key, xy, hand["label"], start)
            temporal.update(
                {
                    "track": key,
                    "handedness": hand["handedness"],
                    "score": round(hand["score"], 3),
                    "points": xy.tolist(),
                }
            )
            hand_results.append(temporal)
        if len(hands) > 1:
            self.gestures = GestureTracker()
        self.gestures.prune(start)
        for store in [self.hand_positions, self.face_positions]:
            for key in list(store):
                if start - store[key][1] > 1.0:
                    del store[key]
        self.blinks = {
            key: value
            for key, value in self.blinks.items()
            if key in self.face_positions
        }
        primary = faces[0] if len(faces) == 1 else None
        duration = (time.monotonic() - start) * 1000
        state = self.controller.update(
            primary["identity"] if primary else None,
            bool(primary and primary["facing"] and not self.enrollment),
            len(faces),
            hand_results,
            start,
            fresh=duration < 500,
        )
        for face in faces:
            del face["embedding"]
        self.frames += 1
        self.frame_times.append(time.monotonic())
        self.processing_ms.append(duration)
        elapsed = self.frame_times[-1] - self.frame_times[0]
        rate = (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0
        trial_status = self.update_trial(hand_results, start)
        response = {
            "faces": faces,
            "hands": hand_results,
            "presentation": state,
            "message": message,
            "enrolling": self.enrollment is not None,
            "profiles": self.profiles.names(),
            "analysis_fps": round(rate, 1),
            "processing_ms": round(duration, 1),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "trial": trial_status,
        }
        self.last_summary = response
        return response

    def start_trial(self, expected: str) -> None:
        allowed = {
            "Open Palm",
            "Fist",
            "Thumb Up",
            "Peace",
            "Point",
            "Swipe Left",
            "Swipe Right",
            "Pinch",
            "No hand",
            "Thumb Left",
            "Thumb Right",
        }
        if expected not in allowed:
            raise ValueError("Choose a supported trial label.")
        self.trial = {
            "expected": expected,
            "start": time.monotonic(),
            "labels": [],
            "events": [],
        }

    def update_trial(self, hands: list[dict], now: float) -> str:
        if self.trial is None:
            return f"{len(self.trials)} completed trials"
        trial = self.trial
        elapsed = now - trial["start"]
        if elapsed >= 1:
            trial["labels"].append(
                hands[0]["static"]
                if len(hands) == 1
                else "No hand"
                if not hands
                else "Multiple hands"
            )
            trial["events"].extend(h["event"] for h in hands if h["event"])
        if elapsed < 4:
            return f"{trial['expected']}: {max(0, 4 - elapsed):.1f}s remaining"
        dynamic = trial["expected"] in {"Swipe Left", "Swipe Right", "Pinch"}
        observations = trial["events"] if dynamic else trial["labels"]
        predicted = (
            Counter(observations).most_common(1)[0][0] if observations else "No event"
        )
        record = {
            "expected": trial["expected"],
            "predicted": predicted,
            "correct": predicted == trial["expected"],
            "analyzed_frames": len(trial["labels"]),
            "event_count": len(trial["events"]),
        }
        self.trials.append(record)
        self.trial = None
        return f"Trial: expected {record['expected']}, observed {predicted}"

    def results(self) -> dict:
        values = list(self.processing_ms)
        return {
            "frames_analyzed": self.frames,
            "profiles_enrolled": len(self.profiles.names()),
            "processing_median_ms": round(float(np.median(values)), 2)
            if values
            else None,
            "processing_p95_ms": round(float(np.percentile(values, 95)), 2)
            if values
            else None,
            "trials": self.trials,
            "actions": list(self.controller.log),
            "note": "Trials are user-labelled live tests. No images or face embeddings are included.",
        }

    def stop(self) -> None:
        """Pause tracking while preserving completed trials and measured timings."""
        self.controller.armed = False
        self.controller.identity = None
        self.enrollment = None
        self.trial = None
        self.last_sequence = -1
        self.last_frame = -100.0
        self.gestures = GestureTracker()
        self.blinks = {}
        self.orientation = PresenterOrientation()
        self.orientation_owner = None
        self.hand_positions = {}
        self.face_positions = {}
        self.frame_times.clear()


def launch_interface(application: PresenterApplication, html: str, port: int = 8765):
    """Start a loopback server in a daemon thread; opening it does not start a camera."""
    token = secrets.token_urlsafe(24)
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(request):
        supplied = request.query_params.get("token", "")
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(403, "Invalid session token.")

    @api.get("/")
    async def home(request: Request):
        authorize(request)
        return HTMLResponse(html.replace("__TOKEN__", token))

    @api.post("/frame")
    async def frame(request: Request):
        authorize(request)
        payload = await request.body()
        if len(payload) > 1_500_000:
            raise HTTPException(413, "Frame too large.")
        if not application.lock.acquire(blocking=False):
            raise HTTPException(409, "Processor busy.")
        try:
            image = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError("Invalid JPEG frame.")
            sequence = int(request.query_params.get("sequence", "-1"))
            result = await asyncio.to_thread(application.process, image, sequence)
            return JSONResponse(result)
        except (ValueError, cv2.error) as exc:
            raise HTTPException(400, str(exc))
        finally:
            application.lock.release()

    @api.post("/action")
    async def action(request: Request):
        authorize(request)
        body = await request.json()
        if not application.lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for the current frame.")
        try:
            kind = body.get("action")
            if kind == "enroll":
                message = application.request_enrollment(
                    str(body.get("name", "")), body.get("consent") is True
                )
            elif kind == "reset":
                application.reset()
                message = "Session and profiles reset."
            elif kind == "stop":
                application.stop()
                message = "Camera stopped; profiles and test results retained."
            elif kind == "trial":
                application.start_trial(str(body.get("expected", "")))
                message = "Trial starts after one second."
            else:
                raise ValueError("Unknown action.")
            return {"message": message}
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        finally:
            application.lock.release()

    @api.get("/results")
    async def results(request: Request):
        authorize(request)
        if not application.lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for the current frame.")
        try:
            return application.results()
        finally:
            application.lock.release()

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", port))
        port = listener.getsockname()[1]
    except OSError:
        listener.close()
        raise
    server = uvicorn.Server(
        uvicorn.Config(
            api, host="127.0.0.1", port=port, log_level="warning", access_log=False
        )
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        listener.close()
        raise RuntimeError("The local interface did not start.")
    return server, thread, f"http://127.0.0.1:{port}/?token={token}"

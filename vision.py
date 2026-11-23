"""Local pretrained models; all required assets are downloaded and hash-checked."""

from pathlib import Path
import hashlib
import os
import struct
import tempfile
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np


def model_assets(manifest: dict, cache: Path | None = None) -> dict[str, Path]:
    """Fetch pinned model files atomically. No pre-existing local input is needed."""
    if cache is None:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
        cache = (
            (Path(base) if base else Path.home() / ".cache")
            / "palm-presenter"
            / "models"
        )
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, spec in manifest.items():
        target = cache / name
        valid = (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == spec["sha256"]
        )
        if not valid:
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "PalmPresenter/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if (
                len(data) != spec["bytes"]
                or hashlib.sha256(data).hexdigest() != spec["sha256"]
            ):
                raise ValueError(f"Model download failed integrity check: {name}")
            with tempfile.NamedTemporaryFile(
                dir=cache, suffix=".part", delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                temporary.write_bytes(data)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        paths[name] = target
    return paths


def caffe_mean(data: bytes) -> np.ndarray:
    """Read the author's Caffe BlobProto mean without installing Caffe.

    Only integer dimensions and packed/unpacked float data are required.
    Unknown protobuf wire types are rejected rather than silently misread.
    """
    position = 0
    fields = {}
    values = []

    def varint():
        nonlocal position
        value = shift = 0
        while position < len(data) and shift <= 63:
            byte = data[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7
        raise ValueError("Malformed mean protobuf.")

    while position < len(data):
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            fields[field] = varint()
        elif wire == 2:
            length = varint()
            payload = data[position : position + length]
            position += length
            if field == 5:
                values.extend(np.frombuffer(payload, dtype="<f4"))
        elif wire == 5:
            value = struct.unpack_from("<f", data, position)[0]
            position += 4
            if field == 5:
                values.append(value)
        elif wire == 1:
            position += 8
        else:
            raise ValueError("Unsupported mean protobuf field.")
    shape = (fields.get(2, 3), fields.get(3, 256), fields.get(4, 256))
    mean = np.asarray(values, dtype=np.float32).reshape(shape)
    if mean.shape != (3, 256, 256):
        raise ValueError("Unexpected mean image shape.")
    return mean[:, 14:241, 14:241]


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Six ordered eye points: two vertical distances divided by twice width."""
    points = np.asarray(points, dtype=float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid eye points.")
    width = np.linalg.norm(points[0] - points[3])
    return (
        float(
            (
                np.linalg.norm(points[1] - points[5])
                + np.linalg.norm(points[2] - points[4])
            )
            / (2 * width)
        )
        if width > 1e-6
        else float("nan")
    )


def estimate_pose(points: np.ndarray, image_size: tuple[int, int]) -> dict:
    """Approximate pose from six landmarks and an uncalibrated pinhole camera.

    This is an orientation proxy, not calibrated eye gaze or attention.
    Reprojection error is reported to expose poor model fits.
    """
    width, height = image_size
    model = np.array(
        [
            (0, 0, 0),
            (0, 330, 65),
            (-225, -170, 135),
            (225, -170, 135),
            (-150, 150, 125),
            (150, 150, 125),
        ],
        dtype=np.float64,
    )
    camera = np.array(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64
    )
    ok, rotation, translation = cv2.solvePnP(
        model,
        points.astype(np.float64),
        camera,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok and translation[2, 0] <= 0:
        # The iterative fit can place the face behind the camera.
        ok, rotation, translation = cv2.solvePnP(
            model,
            points.astype(np.float64),
            camera,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_SQPNP,
        )
    if not ok or translation[2, 0] <= 0:
        return {
            "pitch": None,
            "yaw": None,
            "roll": None,
            "error": None,
            "facing": False,
        }
    angles = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
    projection, _ = cv2.projectPoints(
        model, rotation, translation, camera, np.zeros((4, 1))
    )
    error = float(np.linalg.norm(projection.reshape(-1, 2) - points, axis=1).mean())
    pitch, yaw, roll = map(float, angles)
    valid = translation[2, 0] > 0 and error < max(12.0, width * 0.04)
    return {
        "pitch": round(pitch, 1),
        "yaw": round(yaw, 1),
        "roll": round(roll, 1),
        "error": round(error, 2),
        "facing": bool(valid and abs(yaw) < 25 and abs(pitch) < 25),
    }


class VisionModels:
    """YuNet/SFace, a seven-expression CNN, age bands and MediaPipe landmarks."""

    EXPRESSIONS = [
        "Angry expression",
        "Disgust expression",
        "Fearful expression",
        "Happy expression",
        "Neutral expression",
        "Sad expression",
        "Surprised expression",
    ]
    AGE_BANDS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
    STATIC_LABELS = {
        "Open_Palm": "Open Palm",
        "Closed_Fist": "Fist",
        "Thumb_Up": "Thumb Up",
        "Victory": "Peace",
        "Pointing_Up": "Point",
    }

    def __init__(self, paths: dict[str, Path], tracking: bool = False):
        cv2.setNumThreads(4)
        self.tracking = tracking
        self.timestamp = 0
        self.attribute_cache = []
        self.detector = cv2.FaceDetectorYN.create(
            str(paths["yunet.onnx"]), "", (640, 480), 0.8, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(paths["sface.onnx"]), "")
        self.expression = cv2.dnn.readNet(str(paths["expression.onnx"]))
        self.age = cv2.dnn.readNetFromCaffe(
            str(paths["age.prototxt"]), str(paths["age.caffemodel"])
        )
        self.mean = caffe_mean(paths["age_mean.binaryproto"].read_bytes())
        base = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        self.face_mesh = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(paths["face_landmarker.task"])),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
        )
        self.gestures = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=base(
                    model_asset_path=str(paths["gesture_recognizer.task"])
                ),
                running_mode=vision.RunningMode.VIDEO
                if tracking
                else vision.RunningMode.IMAGE,
                num_hands=8,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
        )

    def analyze_faces(self, frame: np.ndarray) -> list[dict]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(frame)
        result = []
        for detection in [] if detections is None else detections:
            x, y, w, h = map(float, detection[:4])
            x1, y1 = max(0, int(x - 0.15 * w)), max(0, int(y - 0.15 * h))
            x2, y2 = min(width, int(x + 1.15 * w)), min(height, int(y + 1.15 * h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            aligned = self.recognizer.alignCrop(frame, detection)
            embedding = self.recognizer.feature(aligned).reshape(-1)
            # Age changes slowly; expressions update at up to 3 Hz. Identity,
            # eyes and pose are still measured on every analyzed frame.
            now = time.monotonic()
            unit = embedding / max(np.linalg.norm(embedding), 1e-8)
            self.attribute_cache = [
                c for c in self.attribute_cache if now - c["seen"] < 3
            ]
            cache = next(
                (c for c in self.attribute_cache if float(c["unit"] @ unit) > 0.65),
                None,
            )
            if cache is None:
                cache = {
                    "unit": unit,
                    "seen": now,
                    "expression_at": -100,
                    "age_at": -100,
                }
                self.attribute_cache.append(cache)
            cache["seen"] = now
            if now - cache["expression_at"] >= 0.33:
                blob = cv2.dnn.blobFromImage(
                    aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
                )
                self.expression.setInput(blob, "data")
                logits = self.expression.forward("label").reshape(-1)
                scores = np.exp(logits - logits.max())
                scores /= scores.sum()
                cache["expression"] = scores
                cache["expression_at"] = now
            if now - cache["age_at"] >= 2:
                age_image = (
                    cv2.resize(crop, (256, 256))[14:241, 14:241]
                    .transpose(2, 0, 1)
                    .astype(np.float32)
                )
                self.age.setInput((age_image - self.mean)[None, ...])
                cache["age"] = self.age.forward().reshape(-1)
                cache["age_at"] = now
            scores = cache["expression"]
            age_scores = cache["age"]
            mesh = self.face_mesh.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                )
            )
            ear = None
            pose = {
                "pitch": None,
                "yaw": None,
                "roll": None,
                "error": None,
                "facing": False,
            }
            if mesh.face_landmarks:
                landmarks = np.array(
                    [
                        (p.x * (x2 - x1) + x1, p.y * (y2 - y1) + y1)
                        for p in mesh.face_landmarks[0]
                    ]
                )
                ear = (
                    eye_aspect_ratio(landmarks[[33, 160, 158, 133, 153, 144]])
                    + eye_aspect_ratio(landmarks[[362, 385, 387, 263, 373, 380]])
                ) / 2
                pose = estimate_pose(
                    landmarks[[1, 152, 33, 263, 61, 291]], (width, height)
                )
            result.append(
                {
                    "box": [x, y, w, h],
                    "detection_score": float(detection[-1]),
                    "embedding": embedding,
                    "expression": self.EXPRESSIONS[int(scores.argmax())],
                    "expression_score": float(scores.max()),
                    "age_band": self.AGE_BANDS[int(age_scores.argmax())],
                    "age_score": float(age_scores.max()),
                    "ear": ear,
                    "pose": pose,
                    "expression_age_ms": round((now - cache["expression_at"]) * 1000),
                    "age_estimate_age_ms": round((now - cache["age_at"]) * 1000),
                }
            )
        return result

    def analyze_hands(self, frame: np.ndarray, mirrored: bool = False) -> list[dict]:
        height, width = frame.shape[:2]
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        self.timestamp = max(self.timestamp + 1, int(time.monotonic() * 1000))
        output = (
            self.gestures.recognize_for_video(image, self.timestamp)
            if self.tracking
            else self.gestures.recognize(image)
        )
        hands = []
        for points, labels, sides in zip(
            output.hand_landmarks, output.gestures, output.handedness
        ):
            category = labels[0]
            xy = np.array([(p.x * width, p.y * height) for p in points])
            label = (
                self.STATIC_LABELS.get(category.category_name, "Other")
                if category.score >= 0.65
                else "Uncertain"
            )
            side = sides[0].category_name
            if mirrored:
                side = {"Left": "Right", "Right": "Left"}.get(side, side)
            hands.append(
                {
                    "points": xy,
                    "label": label,
                    "score": float(category.score),
                    "handedness": side,
                    "handedness_score": float(sides[0].score),
                }
            )
        return hands

    def close(self) -> None:
        self.face_mesh.close()
        self.gestures.close()

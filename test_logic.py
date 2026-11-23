"""Deterministic logic tests; these do not measure webcam recognition accuracy."""

import unittest
import numpy as np
from logic import (
    ProfileDatabase,
    BlinkMeter,
    PresenterOrientation,
    GestureTracker,
    PresentationController,
)
from vision import eye_aspect_ratio, estimate_pose
import cv2


def hand_points(x=0, pinch=False):
    p = np.zeros((21, 2), dtype=float)
    p[:, 0] = np.linspace(0, 60, 21) + x
    p[:, 1] = 100
    p[5] = [x, 100]
    p[17] = [x + 60, 100]
    p[4] = [x + 20, 10]
    p[8] = [x + 25 if pinch else x + 70, 10]
    return p


def gesture(label="Thumb Up", event=None, stable=1):
    return {"static": label, "event": event, "stable_seconds": stable}


class LogicTests(unittest.TestCase):
    def test_empty_database_and_invalid_embeddings(self):
        db = ProfileDatabase()
        v = np.eye(128, dtype=np.float32)[0]
        self.assertEqual(db.match(v), (None, 0.0))
        with self.assertRaises(ValueError):
            db.match(np.zeros(128))
        with self.assertRaises(ValueError):
            db.enroll("A", [v] * 4)

    def test_enrollment_match_unknown_and_clear(self):
        db = ProfileDatabase()
        v, w = np.eye(128, dtype=np.float32)[:2]
        db.enroll("A", [v] * 5)
        self.assertEqual(db.match(v)[0], "A")
        self.assertIsNone(db.match(w)[0])
        db.clear()
        self.assertEqual(db.names(), [])

    def test_ambiguous_profile_is_rejected(self):
        db = ProfileDatabase()
        v = np.eye(128, dtype=np.float32)[0]
        db.enroll("A", [v] * 5)
        db.enroll("B", [v] * 5)
        self.assertIsNone(db.match(v)[0])

    def test_inconsistent_enrollment_rejected(self):
        db = ProfileDatabase()
        v, w = np.eye(128, dtype=np.float32)[:2]
        with self.assertRaises(ValueError):
            db.enroll("A", [v, v, w, v, v])

    def test_completed_blink_and_hysteresis(self):
        m = BlinkMeter()
        for t, e in [(0, 0.3), (0.1, 0.18), (0.2, 0.22), (0.3, 0.3)]:
            s = m.update(e, t)
        self.assertEqual(s["blink_count"], 1)
        self.assertIsNone(s["blink_rate"])

    def test_long_closure_and_missing_frames_are_not_blinks(self):
        m = BlinkMeter()
        for t in np.arange(0, 2, 0.1):
            m.update(0.1, float(t))
        self.assertEqual(m.update(0.3, 2)["blink_count"], 0)
        m.update(0.1, 2.1)
        m.update(None, 2.2)
        self.assertEqual(m.update(0.3, 2.3)["blink_count"], 0)

    def test_blink_rate_uses_observation_time(self):
        m = BlinkMeter()
        for t in np.arange(0, 12, 0.1):
            m.update(0.1 if 1 <= t <= 1.2 else 0.3, float(t))
        self.assertAlmostEqual(m.snapshot(12)["blink_rate"], 5.0, delta=0.1)

    def test_eye_geometry(self):
        p = np.array([[0, 0], [1, -1], [3, -1], [4, 0], [3, 1], [1, 1]])
        self.assertAlmostEqual(eye_aspect_ratio(p), 0.5)

    def test_pose_recovers_known_projection(self):
        model = np.array(
            [
                (0, 0, 0),
                (0, 330, 65),
                (-225, -170, 135),
                (225, -170, 135),
                (-150, 150, 125),
                (150, 150, 125),
            ],
            float,
        )
        camera = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], float)
        pts, _ = cv2.projectPoints(
            model, np.zeros(3), np.array([0.0, 0.0, 1600.0]), camera, np.zeros(4)
        )
        result = estimate_pose(pts.reshape(-1, 2), (640, 480))
        self.assertTrue(result["facing"])
        self.assertLess(abs(result["yaw"]), 0.01)

    def test_pinch_fires_once_until_release(self):
        t = GestureTracker()
        events = [
            t.update("a", hand_points(pinch=True), "Point", n * 0.1)["event"]
            for n in range(10)
        ]
        self.assertEqual(events.count("Pinch"), 1)
        t.update("a", hand_points(), "Point", 1.0)
        self.assertEqual(
            t.update("a", hand_points(pinch=True), "Point", 1.1)["event"], "Pinch"
        )

    def test_swipes_and_stationary_negative(self):
        for direction, expected in [(1, "Swipe Right"), (-1, "Swipe Left")]:
            t = GestureTracker()
            events = []
            for n in range(7):
                events.append(
                    t.update(
                        "a", hand_points(direction * n * 25), "Open Palm", n * 0.08
                    )["event"]
                )
            self.assertEqual(events.count(expected), 1)
        t = GestureTracker()
        self.assertFalse(
            any(
                t.update("a", hand_points(), "Open Palm", n * 0.08)["event"]
                for n in range(10)
            )
        )

    def test_swipe_requires_open_palm_and_continuity(self):
        t = GestureTracker()
        self.assertFalse(
            any(
                t.update("a", hand_points(n * 40), "Point", n * 0.08)["event"]
                for n in range(6)
            )
        )
        t = GestureTracker()
        t.update("a", hand_points(), "Open Palm", 0)
        self.assertIsNone(t.update("a", hand_points(200), "Open Palm", 1)["event"])

    def test_two_hands_do_not_share_history(self):
        t = GestureTracker()
        t.update("a", hand_points(), "Open Palm", 0)
        self.assertIsNone(t.update("b", hand_points(200), "Open Palm", 0.2)["event"])

    def test_swipe_with_uncertain_frame_and_slow_sampling(self):
        for sign, expected in ((1, "Swipe Right"), (-1, "Swipe Left")):
            tracker = GestureTracker()
            steps = [
                (0, "Open Palm", 0),
                (28, "Uncertain", 0.15),
                (76, "Open Palm", 0.36),
            ]
            observed = [
                tracker.update("a", hand_points(sign * x), label, t)["event"]
                for x, label, t in steps
            ]
            self.assertEqual([event for event in observed if event], [expected])

    def test_diagonal_and_small_movements_do_not_advance(self):
        for dx, dy in ((80, 100), (20, 2)):
            tracker = GestureTracker()
            for i in range(4):
                points = hand_points(i * dx / 3)
                points[:, 1] += i * dy / 3
                self.assertIsNone(
                    tracker.update("a", points, "Open Palm", i * 0.15)["event"]
                )

    def test_orientation_hysteresis_and_sustained_turn(self):
        gate = PresenterOrientation()
        self.assertFalse(gate.update(None, None, False, 0))
        self.assertTrue(gate.update(5, 8, True, 0.1))
        self.assertTrue(gate.update(34, 40, True, 0.3))
        self.assertTrue(gate.update(None, None, False, 0.4))
        self.assertTrue(gate.update(32, 48, True, 0.6))
        self.assertFalse(gate.update(32, 48, True, 0.8))
        self.assertFalse(gate.update(34, 40, True, 0.9))
        self.assertTrue(gate.update(4, 6, True, 1))
        self.assertFalse(gate.update(5, 75, True, 1.1))

    def test_sideways_thumb_moves_once_until_deliberate_release(self):
        controller = self.enabled()

        def show(label, t, stable=1):
            controller.update("A", True, 1, [gesture(label, stable=stable)], t)

        show("Thumb Right", 1.4, 0.15)
        self.assertEqual(controller.slide, 0)
        show("Thumb Right", 1.7)
        show("Thumb Right", 2.7)
        show("Uncertain", 2.8)
        show("Thumb Right", 2.9)
        show("Thumb Right", 3.5)
        self.assertEqual(controller.slide, 1)
        show("Open Palm", 3.6)
        show("Open Palm", 4.0)
        show("Thumb Left", 4.6)
        self.assertEqual(controller.slide, 0)
        self.assertTrue(controller.armed)

    def test_sideways_thumb_needs_enabled_presenter(self):
        controller = PresentationController()
        for t in (0, 1, 2):
            controller.update("A", True, 1, [gesture("Thumb Right")], t)
        self.assertFalse(controller.armed)
        self.assertEqual(controller.slide, 0)

    def test_first_swipe_after_enabling(self):
        controller = self.enabled()
        controller.update("A", True, 1, [gesture("Open Palm", "Swipe Left")], 0.7)
        self.assertEqual(controller.slide, 1)
        controller.update("A", True, 1, [gesture("Open Palm", "Swipe Left")], 0.8)
        self.assertEqual(controller.slide, 1)

    def enabled(self):
        c = PresentationController()
        c.update("A", True, 1, [gesture()], 0)
        c.update("A", True, 1, [gesture()], 0.5)
        self.assertTrue(c.armed)
        return c

    def test_unknown_multiple_faces_and_lookaway_block(self):
        for identity, facing, count in [
            (None, True, 1),
            ("A", True, 2),
            ("A", False, 1),
        ]:
            c = PresentationController()
            c.update(identity, facing, count, [gesture()], 0)
            c.update(identity, facing, count, [gesture()], 1)
            self.assertFalse(c.armed)

    def test_dwell_and_two_hands_block_commands(self):
        c = PresentationController()
        c.update("A", True, 1, [gesture(stable=0)], 0)
        c.update("A", True, 1, [gesture(stable=0.1)], 0.5)
        self.assertFalse(c.armed)
        c.update("A", True, 1, [gesture(), gesture()], 1)
        self.assertFalse(c.armed)

    def test_slide_bounds_pause_and_stale(self):
        c = self.enabled()
        for n in range(10):
            c.update("A", True, 1, [gesture("Open Palm", "Swipe Left")], 2 + n)
        self.assertEqual(c.slide, 4)
        c.update("A", True, 1, [gesture("Fist")], 15)
        self.assertFalse(c.armed)
        c = self.enabled()
        c.update("A", True, 1, [gesture("Open Palm", "Swipe Left")], 2, fresh=False)
        self.assertFalse(c.armed)
        self.assertEqual(c.slide, 0)

    def test_held_static_pose_does_not_repeat(self):
        c = self.enabled()
        c.update("A", True, 1, [gesture("Peace")], 2)
        self.assertTrue(c.blackout)
        c.update("A", True, 1, [gesture("Peace")], 4)
        self.assertTrue(c.blackout)
        c.update("A", True, 1, [gesture("Other")], 4.1)
        c.update("A", True, 1, [gesture("Peace")], 5)
        self.assertFalse(c.blackout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

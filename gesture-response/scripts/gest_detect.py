#!/usr/bin/env python3
"""The confirm gate: what the duck has decided the locked person said.

TWO SEPARATE GATES, AND KEEPING THEM SEPARATE IS THE POINT
------------------------------------------------------------
* **Acquisition** decides WHO.  It lives in :mod:`gest_acquire`: an explicit
  ``search -> found -> locked`` walk that only the requested identity can
  satisfy, gated on the real camera for a MEASURED dwell.  Once locked, no
  other person can become the subject for the rest of the session.
* **Confirmation** decides WHAT, and that is this module.  A gesture is acted on
  only after the SAME command has been read from the LOCKED person, with the arm
  FULLY READABLE, for a sustained window.

A single combined gate would let a strong reading from a stranger substitute for
a weak one from the instructor.  Split, the wrong-person case cannot even reach
the confirm logic: :meth:`GestureDetector.feed` never scores a body that is not
the locked subject, and the metrics record every distractor gesture that WAS
readable and WAS ignored, so the refusal is evidence rather than an absence.

The rolling hand history the motion features come from lives in
:mod:`gest_acquire` too - it is what the duck observed, not what it concluded.

WHAT THIS MODULE MAY NOT IMPORT
---------------------------------
``gest_actors`` and ``gest_script``.  ``tests/test_rollout_and_hygiene.py``
parses the import graph with ``ast`` and fails if it ever does.
"""

from __future__ import annotations

from gest_acquire import Acquisition, Candidate, HandTrack
from gest_cast import BY_NAME
from gest_gesture import classify
from gest_pose import measure_body
from gest_states import (
    CONFIRM_MIN_FRACTION,
    CONFIRM_S,
    GESTURE_MAX_RANGE_M,
    INTERRUPT_COMMAND,
    REJECT_COOLDOWN_S,
)


class GestureDetector:
    """Reads gestures from the locked person, and logs what it refused.

    Every acceptance carries the evidence it was made on: how long the command
    was sustained, what fraction of that window it matched, whether the arm was
    fully readable throughout, and the rule-margin proxy it scored.
    """

    def __init__(self, dt: float, wanted: str):
        self.dt = float(dt)
        self.acquisition = Acquisition(wanted=wanted)
        self.tracks: dict[str, HandTrack] = {}
        self.candidate: Candidate | None = None
        self.accepted: list[dict] = []
        self.rejections: list[dict] = []
        # Everything measured about people who are NOT the locked subject,
        # accumulated so that ignoring them is evidenced rather than assumed.
        self.other_readings: dict[str, dict] = {}
        self._reject_until: dict[str, float] = {}
        self._last_reading = None
        self._suspended = False
        # While the duck is WALKING the detector narrows to the interrupt
        # command rather than shutting down entirely.  See :meth:`suspend`.
        self._interrupt_only = False
        # Every tick on which an interrupt was READ while the duck was walking,
        # so "the STOP was given mid-walk" is a measurement rather than a claim.
        self.interrupt_ticks = 0

    # -- the per-tick measurement -----------------------------------------
    def _track_for(self, name: str) -> HandTrack:
        if name not in self.tracks:
            self.tracks[name] = HandTrack(dt=self.dt)
        return self.tracks[name]

    def read_person(self, name: str, yaw: float, keypoints: dict,
                    body_entry: dict):
        """Measure one person's pose and classify it.  No acceptance here.

        Used for the locked subject AND for everybody else, with the SAME code
        path, which is what makes "a distractor's gesture was fully readable and
        was still ignored" a measurement rather than a claim.
        """
        if not keypoints:
            return None, None, 0.0, 0.0
        readable = body_entry.get("arm_readable", {"l": False, "r": False})
        pose = measure_body(name, yaw, keypoints, readable)
        track = self._track_for(name)
        track.push(keypoints["l_hand"], keypoints["r_hand"])
        travel, wander, full = track.features(BY_NAME[name].arm_span)
        if not full:
            # A partial window under-reports path, which would make a real
            # oscillation look still.  Report the pose but no classification.
            return pose, None, travel, wander
        return pose, classify(pose, travel, wander), travel, wander

    def feed(self, t: float, *, visibility: dict, keypoints: dict,
             yaws: dict, ranges: dict) -> dict:
        """Advance acquisition and the confirm gate by one tick.

        Returns this tick's view: who is locked, what is being confirmed, and
        how far through the confirm window it is.
        """
        locked = self.acquisition.feed(t, self.dt, visibility)
        if self._interrupt_only:
            # The narrowed window is still a real reading path, so the hand
            # history has to keep accumulating for the locked person or the
            # motion features would be empty the moment it opened.
            pass

        # EVERY OTHER PERSON IS MEASURED TOO, and never acted on.  This is what
        # populates the wrong-person evidence: for each distractor the metrics
        # can report the ticks on which their gesture was classifiable AND their
        # arm was fully readable AND they were within gesture range - which is
        # precisely the state in which an identity-blind robot would have obeyed.
        for name, entry in visibility.items():
            if name == locked or not entry.get("present"):
                continue
            pose, reading, travel, wander = self.read_person(
                name, yaws.get(name, 0.0), keypoints.get(name, {}), entry)
            if pose is None or reading is None:
                continue
            record = self.other_readings.setdefault(
                name, {"readable_command_ticks": 0, "commands": {},
                       "max_confidence": 0.0, "first_s": None, "last_s": None,
                       "in_range_ticks": 0, "windows": []})
            in_range = ranges.get(name, 1e9) <= GESTURE_MAX_RANGE_M
            if reading.accepted and pose.fully_readable and in_range:
                record["readable_command_ticks"] += 1
                record["commands"][reading.command] = \
                    record["commands"].get(reading.command, 0) + 1
                record["max_confidence"] = max(
                    record["max_confidence"], reading.confidence)
                if record["first_s"] is None:
                    record["first_s"] = round(float(t), 3)
                record["last_s"] = round(float(t), 3)
                # Contiguous windows, so "sustained past the confirm window" is
                # checkable per gesture rather than as one total.
                if record["windows"] and \
                        t - record["windows"][-1]["to_s"] <= 2.5 * self.dt:
                    record["windows"][-1]["to_s"] = round(float(t), 3)
                    record["windows"][-1]["ticks"] += 1
                else:
                    record["windows"].append(
                        {"command": reading.command,
                         "from_s": round(float(t), 3),
                         "to_s": round(float(t), 3), "ticks": 1})
            if in_range:
                record["in_range_ticks"] += 1

        if not locked or self._suspended:
            self._last_reading = None
            return self._view(t, locked, None, None)

        entry = visibility.get(locked, {})
        pose, reading, travel, wander = self.read_person(
            locked, yaws.get(locked, 0.0), keypoints.get(locked, {}), entry)
        self._last_reading = reading
        if pose is None or reading is None:
            self._drop(t, "the pose could not be measured")
            return self._view(t, locked, pose, reading)

        in_range = ranges.get(locked, 1e9) <= GESTURE_MAX_RANGE_M
        # THE THREE HARD PRECONDITIONS, checked before any accumulation.  Each
        # is a separate reason a gesture is not confirmed, and each is logged as
        # such rather than folded into a single failure.
        if not entry.get("visible"):
            self._drop(t, "the instructor was not visible")
        elif not pose.fully_readable:
            self._drop(t, "the arm was not fully readable in the camera")
        elif not in_range:
            self._drop(t, "the instructor was beyond the gesture range")
        elif not reading.accepted:
            self._note_rejection(t, locked, reading, pose)
            self._drop(t, "no template matched")
        elif self._interrupt_only and reading.command != INTERRUPT_COMMAND:
            # Read clearly, from the right person, and deliberately NOT acted
            # on: the duck is mid-manoeuvre and only an interrupt may land.
            self._drop(t, "mid-command: only an interrupt may be accepted")
        else:
            if self._interrupt_only:
                self.interrupt_ticks += 1
            self._accumulate(t, reading, pose)

        return self._view(t, locked, pose, reading)

    def _accumulate(self, t: float, reading, pose) -> None:
        if self.candidate is None or self.candidate.command != reading.command:
            self.candidate = Candidate(
                command=reading.command, template=reading.template,
                began_at_s=t)
        candidate = self.candidate
        candidate.ticks += 1
        candidate.matching += 1
        candidate.readable_ticks += 1 if pose.fully_readable else 0
        candidate.best_confidence = max(candidate.best_confidence,
                                        reading.confidence)
        candidate.last_confidence = reading.confidence
        candidate.rule = reading.rule
        candidate.features = dict(reading.features)

    def _drop(self, t: float, reason: str) -> None:
        """A tick that did not match.  Tolerated up to the confirm fraction.

        The candidate is NOT discarded on a single bad tick: the head is
        rate-limited and a distractor can cross the sightline, and a gesture the
        person is plainly still making should survive that.  It IS discarded
        once the matching fraction over the window falls below the bar, which is
        what stops a flickering pose from ever confirming.
        """
        if self.candidate is None:
            return
        self.candidate.ticks += 1
        if self.candidate.fraction < CONFIRM_MIN_FRACTION:
            self.candidate = None

    def _note_rejection(self, t: float, name: str, reading, pose) -> None:
        """Log a pose that was seen clearly and matched nothing.

        Cooled down, so a pose hovering at a window edge produces one logged
        rejection rather than fifty.
        """
        if t < self._reject_until.get(name, -1.0):
            return
        if not pose.fully_readable:
            return
        self._reject_until[name] = t + REJECT_COOLDOWN_S
        primary = pose.primary
        self.rejections.append({
            "t": round(float(t), 3),
            "person": name,
            "reason": reading.rule,
            "best_confidence": round(float(reading.confidence), 4),
            "raised_arms": pose.raised_count,
            "fully_readable": bool(pose.fully_readable),
            "extension": (None if primary is None
                          else round(float(primary.extension), 4)),
            "elevation_deg": (None if primary is None
                              else round(float(primary.elevation_deg), 2)),
            "features": {k: (round(float(v), 4) if isinstance(v, float) else v)
                         for k, v in reading.features.items()},
        })

    def confirmed(self, t: float) -> dict | None:
        """The command whose confirm window is complete, if any.

        Requires the sustained hold AND the matching fraction AND that every
        counted tick had a fully readable arm.  All three, because each rules
        out a different way of being wrong.
        """
        candidate = self.candidate
        if candidate is None:
            return None
        if candidate.ticks * self.dt < CONFIRM_S:
            return None
        if candidate.fraction < CONFIRM_MIN_FRACTION:
            return None
        if candidate.readable_ticks < candidate.matching:
            return None
        return {
            "command": candidate.command,
            "template": candidate.template,
            "began_at_s": round(float(candidate.began_at_s), 3),
            "confirmed_at_s": round(float(t), 3),
            "held_s": round(candidate.ticks * self.dt, 3),
            "match_fraction": round(candidate.fraction, 4),
            "readable_fraction": round(
                candidate.readable_ticks / max(candidate.matching, 1), 4),
            "confidence": round(float(candidate.best_confidence), 4),
            "rule": candidate.rule,
            "features": dict(candidate.features),
        }

    def accept(self, record: dict) -> None:
        """Commit a confirmed command and clear the window."""
        self.accepted.append(record)
        self.candidate = None

    def suspend(self, interrupt_only: bool = False) -> None:
        """Stop reading gestures - used while a command is being executed.

        The duck does not take a new order in the middle of carrying one out,
        which is both what a trainee should do and what stops a half-finished
        physical action from being graded against the wrong command.

        ``interrupt_only`` IS THE ONE EXCEPTION, AND IT IS THE ONE THIS WHOLE
        BEHAVIOR TURNS ON.  A STOP that can only be given to a robot already
        standing still is not a stop, it is a formality: the command's entire
        purpose is to interrupt motion that is already under way.  So while the
        duck is WALKING the detector stays live for exactly one command -
        :data:`INTERRUPT_COMMAND` - and every other reading is discarded unread,
        so the duck still cannot be handed a new destination mid-manoeuvre.

        This is deliberately narrower than simply leaving the detector running.
        The full confirm gate still applies - same locked person, arm fully
        readable, sustained for the whole window - so an interrupt costs exactly
        the same evidence as any other command.  It is the SET OF ACCEPTABLE
        COMMANDS that shrinks to one, never the standard of proof.
        """
        self._suspended = not interrupt_only
        self._interrupt_only = bool(interrupt_only)
        self.candidate = None

    def resume(self) -> None:
        self._suspended = False
        self._interrupt_only = False
        self.candidate = None
        # The hand history is cleared with it: a window spanning the gap would
        # blend two different gestures into one motion measurement.
        for track in self.tracks.values():
            track.clear()

    @property
    def suspended(self) -> bool:
        return self._suspended

    @property
    def interrupt_only(self) -> bool:
        return self._interrupt_only

    @property
    def accepted_commands(self) -> list[str]:
        return [entry["command"] for entry in self.accepted]

    def _view(self, t: float, locked: str, pose, reading) -> dict:
        candidate = self.candidate
        return {
            "locked": locked,
            "acquisition_state": self.acquisition.state,
            "pose": None if pose is None else pose.as_record(),
            "reading": None if reading is None else reading.as_record(),
            "candidate_command": "" if candidate is None else candidate.command,
            "candidate_held_s": 0.0 if candidate is None
            else round(candidate.ticks * self.dt, 3),
            "candidate_fraction": 0.0 if candidate is None
            else round(candidate.fraction, 4),
            "candidate_confidence": 0.0 if candidate is None
            else round(float(candidate.last_confidence), 4),
            "confirm_progress": 0.0 if candidate is None else min(
                1.0, (candidate.ticks * self.dt) / CONFIRM_S),
            "suspended": self._suspended,
        }

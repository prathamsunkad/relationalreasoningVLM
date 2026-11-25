"""Conservative full-validation stopping, separate from best-checkpoint selection.

This is an empirical plateau rule, not proof of optimizer convergence. A run
must have six complete post-warmup checks without meaningful cumulative loss
OR accuracy improvement; a recent improving trend also prevents stopping.
Only held-out validation measurements belong here, never training/test losses.
"""
from __future__ import annotations

import copy
import math
from statistics import median


class ConservativePlateau:
    def __init__(self, *, warmup_updates=500, interval=500, patience=6,
                 absolute_delta=0.001, relative_delta=0.001,
                 accuracy_delta=0.002, min_joint_updates=1000,
                 expected_count=1000):
        integer_values = dict(warmup_updates=warmup_updates, interval=interval,
                              patience=patience, min_joint_updates=min_joint_updates,
                              expected_count=expected_count)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in integer_values.values()):
            raise ValueError("Update, interval, patience and count settings must be integers")
        if warmup_updates < 0 or min_joint_updates < 0 or interval < 1 or patience < 6 or expected_count < 1:
            raise ValueError("Invalid conservative plateau settings (patience must be at least six)")
        for name, value in dict(absolute_delta=absolute_delta, relative_delta=relative_delta,
                                accuracy_delta=accuracy_delta).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid " + name)
        if accuracy_delta > 1:
            raise ValueError("Accuracy is a fraction, not a percentage")
        self.config = dict(**integer_values, absolute_delta=float(absolute_delta),
                           relative_delta=float(relative_delta), accuracy_delta=float(accuracy_delta))
        self.observations = []
        self.history = []
        self.loss_anchor = None
        self.accuracy_anchor = None
        self.bad_checks = 0
        self.stopped = False
        self._last_decision = None

    def _loss_delta(self, anchor):
        return max(self.config["absolute_delta"], self.config["relative_delta"] * abs(anchor))

    def update(self, *, update, loss, accuracy, count, complete=True):
        if not complete:
            return dict(stop=self.stopped, eligible=False, reason="incomplete_validation",
                        update=update, count=count, bad_checks=self.bad_checks)
        if count != self.config["expected_count"]:
            raise ValueError("Plateau checks require the entire fixed validation set")
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise ValueError("Invalid update")
        if not math.isfinite(loss) or loss < 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Invalid validation loss or accuracy")
        observation = dict(update=update, loss=float(loss), accuracy=float(accuracy), count=int(count))
        if self.observations:
            previous = self.observations[-1]
            if update < previous["update"]:
                raise ValueError("Validation updates must be monotonic")
            if update == previous["update"]:
                if observation != previous:
                    raise ValueError("Conflicting validation at the same update")
                return copy.deepcopy(self._last_decision)
        if self.stopped:
            raise ValueError("Training continued after the validation stopping decision")
        self.observations.append(observation)
        decision = dict(stop=False, eligible=False, reason="warmup", update=update,
                        count=count, bad_checks=self.bad_checks)
        if update <= self.config["warmup_updates"]:
            self._last_decision = decision
            return copy.deepcopy(decision)
        # Missing evaluations cannot manufacture a sustained plateau. Start a
        # fresh observation window after a gap larger than the planned interval.
        gap = bool(self.history and update - self.history[-1]["update"] > self.config["interval"])
        if not self.history or gap:
            self.history = [observation]
            self.loss_anchor, self.accuracy_anchor = observation["loss"], observation["accuracy"]
            self.bad_checks = 0
            decision.update(eligible=True, reason="new_window" if gap else "first_joint_validation", bad_checks=0)
        else:
            self.history.append(observation)
            loss_improved = self.loss_anchor - loss >= self._loss_delta(self.loss_anchor) - 1e-12
            accuracy_improved = accuracy - self.accuracy_anchor >= self.config["accuracy_delta"] - 1e-12
            if loss_improved:
                self.loss_anchor = float(loss)
            if accuracy_improved:
                self.accuracy_anchor = float(accuracy)
            self.bad_checks = 0 if loss_improved or accuracy_improved else self.bad_checks + 1
            decision.update(eligible=True, reason="improving" if loss_improved or accuracy_improved else "waiting",
                            bad_checks=self.bad_checks)
            if len(self.history) >= 6:
                previous, recent = self.history[-6:-3], self.history[-3:]
                loss_trend = median(x["loss"] for x in previous) - median(x["loss"] for x in recent)
                accuracy_trend = median(x["accuracy"] for x in recent) - median(x["accuracy"] for x in previous)
                trend_improves = (loss_trend >= self._loss_delta(self.loss_anchor) - 1e-12 or
                                  accuracy_trend >= self.config["accuracy_delta"] - 1e-12)
                decision.update(trend_loss_improvement=loss_trend,
                                trend_accuracy_improvement=accuracy_trend,
                                trend_improving=trend_improves)
                enough_joint_training = update - self.config["warmup_updates"] >= self.config["min_joint_updates"]
                if self.bad_checks >= self.config["patience"] and enough_joint_training:
                    if trend_improves:
                        decision["reason"] = "recent_improving_trend"
                    else:
                        self.stopped = True
                        decision.update(stop=True, reason="validation_plateau")
        decision.update(loss_anchor=self.loss_anchor, accuracy_anchor=self.accuracy_anchor)
        self._last_decision = decision
        return copy.deepcopy(decision)

    def state_dict(self):
        # Replaying this tiny history reconstructs all anchors/counters exactly
        # and validates monotonicity, data count and configuration on restore.
        return dict(version=1, config=copy.deepcopy(self.config),
                    observations=copy.deepcopy(self.observations), stopped=self.stopped)

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported plateau-state version")
        result = cls(**state["config"])
        for row in state["observations"]:
            result.update(**row)
        if result.stopped != state.get("stopped"):
            raise ValueError("Inconsistent plateau stopping state")
        return result

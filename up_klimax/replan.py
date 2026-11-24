# Copyright 2026 Intelligent Robotics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replanning and plan repair for hierarchical problems. UPF's own replanner leaves them out."""

from typing import Optional
from warnings import warn

import unified_planning as up
from unified_planning.engines import Engine
from unified_planning.engines.mixins.oneshot_planner import OptimalityGuarantee
from unified_planning.engines.mixins.plan_repairer import PlanRepairerMixin
from unified_planning.engines.mixins.replanner import ReplannerMixin
from unified_planning.engines.results import (
    LogLevel,
    LogMessage,
    PlanGenerationResult,
    PlanGenerationResultStatus,
    ValidationResultStatus,
)
from unified_planning.model.htn import Task

from .engine import KlimaxEngine, KlimaxValidator, credits
from .kind import SUPPORTED_KIND


class KlimaxReplanner(Engine, ReplannerMixin):
    """``Replanner(problem, name="klimax-replanner")``: the problem is kept and changed between
    calls (initial values, goals, actions), and planned again from scratch with Klimax.

    Removing an action also sets aside the methods that have it as a subtask, until an action
    of that name is added again. Options: those of the planner (``timeout``, ``anytime``).
    """

    def __init__(self, problem, error_on_failed_checks=True, **options):
        Engine.__init__(self)
        self.error_on_failed_checks = error_on_failed_checks
        ReplannerMixin.__init__(self, problem, error_on_failed_checks)
        self._planner = KlimaxEngine(**options)
        self._methods = list(self._problem.methods)  # all of them, whatever the actions left

    @property
    def name(self) -> str:
        return "klimax-replanner"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        return SUPPORTED_KIND

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return KlimaxEngine.supports(problem_kind)

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    @staticmethod
    def get_credits(**kwargs):
        return credits

    def _resolve(self, timeout: Optional[float] = None, output_stream=None):
        return self._planner.solve(self._current(), timeout=timeout, output_stream=output_stream)

    def _current(self):
        """The problem with the methods all of whose actions it still has."""
        problem = self._problem.clone()
        names = {a.name for a in problem.actions}
        problem._methods.clear()  # UPF has no way of removing a method
        for m in self._methods:
            if all(isinstance(s.task, Task) or s.task.name in names for s in m.subtasks):
                problem._methods[m.name] = m
        return problem

    def _update_initial_value(self, fluent, value):
        self._problem.set_initial_value(fluent, value)

    def _add_goal(self, goal):
        self._problem.add_goal(goal)

    def _remove_goal(self, goal):
        (wanted,) = self._problem.environment.expression_manager.auto_promote(goal)
        goals = self._problem.goals
        self._problem.clear_goals()
        for g in goals:
            if g is not wanted:
                self._problem.add_goal(g)
        if len(self._problem.goals) == len(goals):
            self._not_found(f"goal to remove not among the goals: {wanted}")

    def _add_action(self, action):
        self._problem.add_action(action)

    def _remove_action(self, name: str):
        actions = self._problem.actions
        self._problem.clear_actions()
        for a in actions:
            if a.name != name:
                self._problem.add_action(a)
        if len(self._problem.actions) == len(actions):
            self._not_found(f"action to remove not among the actions: {name}")

    def _not_found(self, message):
        if self.error_on_failed_checks:
            raise up.exceptions.UPUsageError(message)
        warn(message)


class KlimaxRepairer(Engine, PlanRepairerMixin):
    """``PlanRepairer(name="klimax-repairer")``: a hierarchical plan that is still valid for
    the problem comes back as it is; otherwise the problem is planned again. Options: those of
    the planner (``timeout``, ``anytime``)."""

    def __init__(self, **options):
        Engine.__init__(self)
        PlanRepairerMixin.__init__(self)
        self._planner = KlimaxEngine(**options)

    @property
    def name(self) -> str:
        return "klimax-repairer"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        return SUPPORTED_KIND

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return KlimaxEngine.supports(problem_kind)

    @staticmethod
    def supports_plan(plan_kind: "up.plans.PlanKind") -> bool:
        return plan_kind == up.plans.PlanKind.HIERARCHICAL_PLAN

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    @staticmethod
    def get_credits(**kwargs):
        return credits

    def _repair(self, problem, plan) -> PlanGenerationResult:
        checked = KlimaxValidator().validate(problem, plan)
        if checked.status == ValidationResultStatus.VALID:
            return PlanGenerationResult(
                PlanGenerationResultStatus.SOLVED_SATISFICING, plan, self.name,
                log_messages=[LogMessage(LogLevel.INFO, "the plan is valid as it is")])
        result = self._planner.solve(problem)
        return PlanGenerationResult(
            result.status, result.plan, self.name, result.metrics,
            [LogMessage(LogLevel.INFO, "the plan was not valid: planned again")] +
            list(result.log_messages or []))

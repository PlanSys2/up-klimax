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

"""Replanning and plan repair of hierarchical problems."""

import pytest
from unified_planning.engines import PlanGenerationResultStatus as S
from unified_planning.engines import ValidationResultStatus as V
from unified_planning.plans import SequentialPlan
from unified_planning.plans.hierarchical_plan import HierarchicalPlan
from unified_planning.shortcuts import PlanRepairer, PlanValidator, Replanner, get_environment

import up_klimax
from test_engine import delivery

up_klimax.register()
get_environment().credits_stream = None


def test_replanning_after_changes():
    problem, _ = delivery()
    at, parcel_in = problem.fluent("at"), problem.fluent("in")
    r0, r2 = problem.object("r0"), problem.object("r2")
    box = problem.object("box")
    with Replanner(problem, name="klimax-replanner") as replanner:
        first = replanner.resolve(timeout=10)
        assert first.status == S.SOLVED_SATISFICING
        assert len(first.plan.action_plan.actions) == 13
        # The robot is elsewhere: the plan starts from there
        replanner.update_initial_value(at(r0), False)
        replanner.update_initial_value(at(r2), True)
        moved = replanner.resolve(timeout=10)
        assert moved.status == S.SOLVED_SATISFICING
        assert str(moved.plan.action_plan.actions[0]) == "move(r2, r3)"
        assert len(moved.plan.action_plan.actions) == 11
        # A goal the tasks cannot leave true, then without it again
        goal = parcel_in(box, r2)
        replanner.add_goal(goal)
        assert replanner.resolve(timeout=10).status == S.UNSOLVABLE_PROVEN
        replanner.remove_goal(goal)
        assert replanner.resolve(timeout=10).status == S.SOLVED_SATISFICING
        # Without `drop`, nothing can be delivered; with it back, it can
        drop = problem.action("drop")
        replanner.remove_action("drop")
        assert replanner.resolve(timeout=10).status == S.UNSOLVABLE_PROVEN
        replanner.add_action(drop)
        assert replanner.resolve(timeout=10).status == S.SOLVED_SATISFICING
        with pytest.warns(UserWarning, match="not among the actions"):
            replanner.remove_action("fly")  # there is none: said, as UPF's checks are set
    # The problem given is not touched
    assert problem.initial_value(at(r0)).is_true() and len(problem.actions) == 3


def test_repair():
    problem, _ = delivery()
    with Replanner(problem, name="klimax-replanner") as replanner:
        plan = replanner.resolve(timeout=10).plan
    with PlanRepairer(name="klimax-repairer") as repairer:
        # Valid: the same plan
        same = repairer.repair(problem, plan)
        assert same.status == S.SOLVED_SATISFICING and same.plan is plan
        # The world changed under the plan: another one, valid for the problem as it is now
        at = problem.fluent("at")
        changed = problem.clone()
        changed.set_initial_value(at(problem.object("r0")), False)
        changed.set_initial_value(at(problem.object("r1")), True)
        repaired = repairer.repair(changed, plan)
        assert repaired.status == S.SOLVED_SATISFICING and repaired.plan is not plan
        with PlanValidator(name="klimax-val") as validator:
            assert validator.validate(changed, plan).status == V.INVALID
            assert validator.validate(changed, repaired.plan).status == V.VALID
        # Broken beyond repair by reordering alone
        actions = plan.action_plan.actions
        broken = HierarchicalPlan(
            SequentialPlan([actions[1], actions[0]] + actions[2:]), plan.decomposition)
        assert repairer.repair(problem, broken).status == S.SOLVED_SATISFICING


def test_chosen_by_upf():
    problem, _ = delivery()
    with Replanner(problem) as replanner:
        assert replanner.name == "klimax-replanner"

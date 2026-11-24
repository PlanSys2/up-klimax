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

"""Every feature Klimax declares, in a problem that it solves and whose plan it verifies."""

from unified_planning.engines import PlanGenerationResultStatus as S
from unified_planning.engines import ValidationResultStatus as V
from unified_planning.model.htn import HierarchicalProblem, Method
from unified_planning.shortcuts import (
    And, BoolType, Equals, Exists, Fluent, Forall, InstantaneousAction,
    MinimizeSequentialPlanLength, Not, Object, OneshotPlanner, Or, PlanValidator, UserType, Variable, get_environment)
from unified_planning.test.examples import get_example_problems

import up_klimax
from up_klimax.kind import SUPPORTED_KIND

up_klimax.register()
get_environment().credits_stream = None


def lamps(variant):
    """Lamps to switch on, by a task with one method; `variant` adds one feature to it."""
    Thing = UserType("Thing")
    Lamp = UserType("Lamp", Thing if variant == "subtypes" else None)
    on = Fluent("on", BoolType(), l=Lamp)
    broken = Fluent("broken", BoolType(), l=Lamp)
    p = HierarchicalProblem(variant)
    p.add_fluent(on, default_initial_value=False)
    p.add_fluent(broken, default_initial_value=False)
    a, b, c = (Object(n, Lamp) for n in "abc")
    p.add_objects([a, b, c])
    switch = InstantaneousAction("switch", l=Lamp)
    x = Variable("x", Lamp)
    if variant == "negative":
        switch.add_precondition(Not(on(switch.l)))
    elif variant == "disjunctive":
        switch.add_precondition(Or(Not(on(switch.l)), broken(switch.l)))
    elif variant == "equalities":
        switch.add_precondition(Not(Equals(switch.l, c)))
    elif variant == "existential":
        switch.add_precondition(Exists(And(Not(on(x)), Not(broken(x))), x))
    elif variant == "universal":
        switch.add_precondition(Forall(Not(broken(x)), x))
    switch.add_effect(on(switch.l), True)
    if variant == "conditional":
        switch.add_effect(broken(switch.l), True, condition=on(switch.l))
    elif variant == "forall":
        switch.add_effect(broken(x), False, forall=[x])
    p.add_action(switch)
    light = p.add_task("light", l=Lamp)
    m = Method("light-it", l=Lamp)
    m.set_task(light, m.l)
    m.add_precondition(Not(broken(m.l)))
    m.add_subtask(switch, m.l)
    p.add_method(m)
    t1 = p.task_network.add_subtask(light, a)
    t2 = p.task_network.add_subtask(light, b)
    if variant != "partial":
        p.task_network.set_ordered(t1, t2)
    p.add_goal(on(a))
    return p


VARIANTS = ["plain", "partial", "subtypes", "negative", "disjunctive", "equalities",
            "existential", "universal", "conditional", "forall"]


def test_each_feature_is_planned_with():
    shortest = lamps("plain")
    shortest.name = "shortest"
    shortest.add_quality_metric(MinimizeSequentialPlanLength())
    problems = [lamps(v) for v in VARIANTS] + [
        get_example_problems()["htn-go"].problem, shortest]
    seen = set()
    for problem in problems:
        assert up_klimax.KlimaxEngine.supports(problem.kind), problem.name
        with OneshotPlanner(name="klimax") as planner:
            result = planner.solve(problem, timeout=10)
        assert result.status in (S.SOLVED_SATISFICING, S.SOLVED_OPTIMALLY), problem.name
        with PlanValidator(name="klimax-val") as validator:
            assert validator.validate(problem, result.plan).status == V.VALID, problem.name
        seen |= problem.kind.features
    assert SUPPORTED_KIND.features - seen == set()

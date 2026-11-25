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
from fractions import Fraction

from unified_planning.shortcuts import (
    GE, LT, And, BoolType, Equals, Exists, Fluent, Forall, InstantaneousAction, IntType,
    MinimizeSequentialPlanLength, Not, Object, OneshotPlanner, Or, PlanValidator, RealType,
    Replanner, UserType, Variable, get_environment)
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


def trip(variant):
    """A truck from a to d along a road, with a tank of 5 that it may fill where there is a
    station; `variant` says how the numbers are written."""
    Place = UserType("Place")
    at = Fluent("at", BoolType(), l=Place)
    road = Fluent("road", BoolType(), a=Place, b=Place)
    station = Fluent("station", BoolType(), l=Place)
    number = RealType() if variant == "real" else IntType()
    fuel = Fluent("fuel", number)
    tank = Fluent("tank", number)
    need = Fluent("need", number, a=Place, b=Place)
    trips = Fluent("trips", IntType())
    p = HierarchicalProblem("trip-" + variant)
    for f in (at, road, station):
        p.add_fluent(f, default_initial_value=False)
    p.add_fluent(fuel, default_initial_value=5)
    p.add_fluent(tank, default_initial_value=5)
    p.add_fluent(need, default_initial_value=1)
    p.add_fluent(trips, default_initial_value=0)
    a, b, c, d = (Object(n, Place) for n in "abcd")
    p.add_objects([a, b, c, d])
    move = InstantaneousAction("move", a=Place, b=Place)
    move.add_precondition(at(move.a))
    move.add_precondition(road(move.a, move.b))
    move.add_effect(at(move.a), False)
    move.add_effect(at(move.b), True)
    if variant == "simple":  # only constants next to the fluents
        move.add_precondition(GE(fuel(), 3))
        move.add_decrease_effect(fuel(), 3)
    else:
        move.add_precondition(GE(fuel(), need(move.a, move.b)))
        move.add_decrease_effect(fuel(), need(move.a, move.b))
    if variant in ("increase", "simple"):
        move.add_increase_effect(trips(), 1)
    fill = InstantaneousAction("fill", l=Place)
    fill.add_precondition(at(fill.l))
    fill.add_precondition(station(fill.l))
    fill.add_precondition(LT(fuel(), 5))
    fill.add_effect(fuel(), tank() if variant == "static" else 5)
    p.add_action(move)
    p.add_action(fill)
    go = p.add_task("go", to=Place)
    there = Method("there", to=Place)
    there.set_task(go, there.to)
    there.add_precondition(at(there.to))
    p.add_method(there)
    step = Method("step", to=Place, a=Place, b=Place)
    step.set_task(go, step.to)
    step.add_precondition(at(step.a))
    step.set_ordered(step.add_subtask(move, step.a, step.b), step.add_subtask(go, step.to))
    p.add_method(step)
    refuel = Method("refuel", to=Place, l=Place)
    refuel.set_task(go, refuel.to)
    refuel.add_precondition(LT(fuel(), 3))
    refuel.set_ordered(refuel.add_subtask(fill, refuel.l), refuel.add_subtask(go, refuel.to))
    p.add_method(refuel)
    p.task_network.add_subtask(go, d)
    half = Fraction(1, 2) if variant == "real" else 0
    for x, y, n in ((a, b, 3), (b, c, 2 + half), (c, d, 3 - half)):
        p.set_initial_value(road(x, y), True)
        p.set_initial_value(need(x, y), n)
    p.set_initial_value(at(a), True)
    for x in (b, c) if variant in ("simple", "shortest") else (b,):
        p.set_initial_value(station(x), True)
    if variant in ("increase", "simple"):
        p.add_goal(Equals(trips(), 3))
    if variant == "shortest":
        p.add_quality_metric(MinimizeSequentialPlanLength())
    return p


NUMBERS = ["int", "real", "increase", "static", "simple", "shortest"]

VARIANTS = ["plain", "partial", "subtypes", "negative", "disjunctive", "equalities",
            "existential", "universal", "conditional", "forall"]


def test_each_feature_is_planned_with():
    shortest = lamps("plain")
    shortest.name = "shortest"
    shortest.add_quality_metric(MinimizeSequentialPlanLength())
    problems = [lamps(v) for v in VARIANTS] + [trip(v) for v in NUMBERS] + [
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
    # UPF 1.3.0 never marks a problem with STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS (it looks for
    # expressions in a set of fluents, so every fluent counts as not static; later versions
    # do): trip("static") is the problem that has it
    assert SUPPORTED_KIND.features - seen <= {"STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS"}


def test_numbers_decide_the_plan():
    # 5 - 3 = 2 at b: enough to reach c, not d from there, so it fills up on the way
    problem = trip("int")
    with OneshotPlanner(name="klimax") as planner:
        result = planner.solve(problem, timeout=10)
    plan = [str(a) for a in result.plan.action_plan.actions]
    assert plan == ["move(a, b)", "fill(b)", "move(b, c)", "move(c, d)"]
    # The shortest, proved, with two stations to choose from
    with OneshotPlanner(name="klimax") as planner:
        result = planner.solve(trip("shortest"), timeout=10)
    assert result.status == S.SOLVED_OPTIMALLY
    assert len(result.plan.action_plan.actions) == 4


def test_no_fuel_no_plan():
    problem = trip("int")
    station = problem.fluent("station")
    problem.set_initial_value(station(problem.object("b")), False)
    with OneshotPlanner(name="klimax") as planner:
        assert planner.solve(problem, timeout=10).status == S.UNSOLVABLE_PROVEN


def test_the_validator_counts_the_fuel():
    # The plan of the problem with a station at b, checked where b has none
    problem = trip("int")
    with OneshotPlanner(name="klimax") as planner:
        plan = planner.solve(problem, timeout=10).plan
    station = problem.fluent("station")
    problem.set_initial_value(station(problem.object("b")), False)
    with PlanValidator(name="klimax-val") as validator:
        assert validator.validate(problem, plan).status == V.INVALID


def test_numbers_in_the_replanner():
    # With less fuel at the start the same trip needs the station sooner, or cannot be made
    problem = trip("int")
    fuel, station = problem.fluent("fuel"), problem.fluent("station")
    with Replanner(problem, name="klimax-replanner") as replanner:
        assert len(replanner.resolve(timeout=10).plan.action_plan.actions) == 4
        replanner.update_initial_value(station(problem.object("a")), True)
        replanner.update_initial_value(fuel(), 2)
        plan = replanner.resolve(timeout=10).plan
        assert [str(a) for a in plan.action_plan.actions][0] == "fill(a)"
        replanner.update_initial_value(station(problem.object("a")), False)
        assert replanner.resolve(timeout=10).status == S.UNSOLVABLE_PROVEN

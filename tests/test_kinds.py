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

from unified_planning.plans import TimeTriggeredPlan
from unified_planning.shortcuts import (
    GE, LT, And, AnytimePlanner, DurativeAction, EndTiming, GlobalStartTiming, StartTiming, BoolType, Equals, Exists, Fluent, Forall, InstantaneousAction,
    Int, IntType, MaximizeExpressionOnFinalState, MinimizeActionCosts,
    MinimizeExpressionOnFinalState, MinimizeSequentialPlanLength, Not, Object, OneshotPlanner,
    Or, PlanValidator, RealType, Replanner, UserType, Variable, get_environment)
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


def roads(variant):
    """From a to d: straight (10), through e (2 + 3) or through b and c (1 + 1 + 1). `variant`
    says what makes a plan better."""
    Place = UserType("Place")
    at = Fluent("at", BoolType(), l=Place)
    road = Fluent("road", BoolType(), a=Place, b=Place)
    length = Fluent("length", RealType() if variant == "real" else IntType(), a=Place, b=Place)
    km = Fluent("km", IntType())
    wear = Fluent("wear", IntType())
    p = HierarchicalProblem("roads-" + variant)
    for f in (at, road):
        p.add_fluent(f, default_initial_value=False)
    p.add_fluent(length, default_initial_value=1)
    p.add_fluent(km, default_initial_value=0)
    p.add_fluent(wear, default_initial_value=1)
    a, b, c, d, e = (Object(n, Place) for n in "abcde")
    p.add_objects([a, b, c, d, e])
    drive = InstantaneousAction("drive", a=Place, b=Place)
    drive.add_precondition(at(drive.a))
    drive.add_precondition(road(drive.a, drive.b))
    drive.add_effect(at(drive.a), False)
    drive.add_effect(at(drive.b), True)
    if variant in ("final", "final-max"):
        drive.add_increase_effect(km(), length(drive.a, drive.b))
    if variant == "state":  # each stretch costs more than the one before
        drive.add_increase_effect(wear(), 1)
    p.add_action(drive)
    go = p.add_task("go", to=Place)
    there = Method("there", to=Place)
    there.set_task(go, there.to)
    there.add_precondition(at(there.to))
    p.add_method(there)
    step = Method("step", to=Place, a=Place, b=Place)
    step.set_task(go, step.to)
    step.add_precondition(at(step.a))
    step.set_ordered(step.add_subtask(drive, step.a, step.b), step.add_subtask(go, step.to))
    p.add_method(step)
    p.task_network.add_subtask(go, d)
    half = Fraction(1, 2) if variant == "real" else 0
    for x, y, n in ((a, d, 10), (a, b, 1 + half), (b, c, 1), (c, d, 1), (a, e, 2), (e, d, 3)):
        p.set_initial_value(road(x, y), True)
        p.set_initial_value(length(x, y), n)
    p.set_initial_value(at(a), True)
    if variant in ("costs", "real"):
        p.add_quality_metric(MinimizeActionCosts({drive: length(drive.a, drive.b)}))
    elif variant == "constant":
        p.add_quality_metric(MinimizeActionCosts({}, default=Int(2)))
    elif variant == "state":
        p.add_quality_metric(MinimizeActionCosts({drive: wear()}))
    elif variant == "final":
        p.add_quality_metric(MinimizeExpressionOnFinalState(km()))
    elif variant == "final-max":
        p.add_quality_metric(MaximizeExpressionOnFinalState(km()))
    return p


def jobs(variant):
    """Two workers with a job each, which lasts; `variant` says what of time the problem has."""
    Worker = UserType("Worker")
    idle = Fluent("idle", BoolType(), w=Worker)
    done = Fluent("done", BoolType(), w=Worker)
    awake = Fluent("awake", BoolType(), w=Worker)
    number = RealType() if variant == "real" else IntType()
    pace = Fluent("pace", number, w=Worker)
    tired = Fluent("tired", IntType())
    p = HierarchicalProblem("jobs-" + variant)
    p.add_fluent(idle, default_initial_value=True)
    p.add_fluent(done, default_initial_value=False)
    p.add_fluent(awake, default_initial_value=variant != "timed")
    p.add_fluent(pace, default_initial_value=5)
    p.add_fluent(tired, default_initial_value=0)
    ann, bob = Object("ann", Worker), Object("bob", Worker)
    p.add_objects([ann, bob])
    work = DurativeAction("work", w=Worker)
    if variant == "between":  # any duration from 3 to 5: Klimax takes the least
        work.set_closed_duration_interval(3, 5)
    elif variant in ("static", "real"):
        work.set_fixed_duration(pace(work.w))
    elif variant == "state":  # each job longer than the one before
        work.set_fixed_duration(pace(work.w) + tired())
        work.add_increase_effect(EndTiming(), tired(), 1)
    else:
        work.set_fixed_duration(5)
    work.add_condition(StartTiming(), idle(work.w))
    work.add_condition(StartTiming(), awake(work.w))
    work.add_effect(StartTiming(), idle(work.w), False)
    work.add_effect(EndTiming(), idle(work.w), True)
    work.add_effect(EndTiming(), done(work.w), True)
    p.add_action(work)
    job = p.add_task("job", w=Worker)
    m = Method("m", w=Worker)
    m.set_task(job, m.w)
    m.add_subtask(work, m.w)
    p.add_method(m)
    t1 = p.task_network.add_subtask(job, ann)
    t2 = p.task_network.add_subtask(job, bob)
    if variant == "ordered":
        p.task_network.set_ordered(t1, t2)
    if variant == "real":
        p.set_initial_value(pace(bob), Fraction(5, 2))
    if variant == "timed":  # nobody is awake before 7
        p.add_timed_effect(GlobalStartTiming(7), awake(ann), True)
        p.add_timed_effect(GlobalStartTiming(7), awake(bob), True)
    p.add_goal(done(ann))
    p.add_goal(done(bob))
    return p


# When the plan ends
TIMES = {"plain": 5, "ordered": 10, "between": 3, "static": 5, "real": 5, "timed": None,
         "state": None}


# What the best plan is worth, and whether Klimax proves it the best
METRICS = {"costs": (3, True), "real": (Fraction(7, 2), True), "constant": (2, True),
           "state": (1, False), "final": (3, True), "final-max": (10, False)}

VARIANTS = ["plain", "partial", "subtypes", "negative", "disjunctive", "equalities",
            "existential", "universal", "conditional", "forall"]


def test_each_feature_is_planned_with():
    shortest = lamps("plain")
    shortest.name = "shortest"
    shortest.add_quality_metric(MinimizeSequentialPlanLength())
    problems = [lamps(v) for v in VARIANTS] + [trip(v) for v in NUMBERS] + [
        roads(v) for v in METRICS] + [jobs(v) for v in TIMES] + [
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
    # Fluents in numeric assignments: UPF 1.3.0 marks every one as not static (it looks for
    # expressions in a set of fluents), later versions mark trip("static")'s as static. One of
    # the two, whichever the version says
    pair = {"STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS", "FLUENTS_IN_NUMERIC_ASSIGNMENTS"}
    assert seen & pair
    assert SUPPORTED_KIND.features - seen <= pair


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


def test_the_metric_decides_the_plan():
    for variant, (best, proved) in METRICS.items():
        problem = roads(variant)
        with OneshotPlanner(name="klimax") as planner:
            result = planner.solve(problem, timeout=10)
        assert result.status == (S.SOLVED_OPTIMALLY if proved else S.SOLVED_SATISFICING), variant
        assert Fraction(result.metrics["klimax_metric"]) == best, variant
        # The validator values the plan the same, by its own means
        with PlanValidator(name="klimax-val") as validator:
            checked = validator.validate(problem, result.plan)
        assert checked.status == V.VALID
        assert checked.metric_evaluations == {problem.quality_metrics[0]: best}, variant
    # The cheapest way is the longest; without a timeout, the first plan found
    with OneshotPlanner(name="klimax") as planner:
        result = planner.solve(roads("costs"), timeout=10)
        assert [str(a) for a in result.plan.action_plan.actions] == [
            "drive(a, b)", "drive(b, c)", "drive(c, d)"]
        assert planner.solve(roads("costs")).status == S.SOLVED_SATISFICING


def test_actions_with_a_cost_and_no_effects():
    # As the optimal cases of UPF's engine tests: actions that only cost. The cheapest way is the
    # longest, and its cost is what the validator says (UPF 1.3's writer left such actions free)
    from unified_planning.model.metrics import MinimizeActionCosts
    from unified_planning.shortcuts import InstantaneousAction, Int
    from unified_planning.model.htn import Task
    problem = HierarchicalProblem("costs_only")
    acts = [InstantaneousAction(f"a{i}") for i in range(5)]
    for act in acts:
        problem.add_action(act)
    task = Task("work")
    problem.add_task(task)
    for name, sub in (("short", acts[:2]), ("long", acts[2:])):
        m = Method(name)
        m.set_task(task)
        m.set_ordered(*[m.add_subtask(a) for a in sub])
        problem.add_method(m)
    problem.task_network.add_subtask(task)
    problem.add_quality_metric(
        MinimizeActionCosts({a: Int(c) for a, c in zip(acts, (1, 10, 1, 1, 1))}))
    with OneshotPlanner(name="klimax") as planner:
        result = planner.solve(problem, timeout=10)
    assert result.status == S.SOLVED_OPTIMALLY
    assert [str(a) for a in result.plan.action_plan.actions] == ["a2", "a3", "a4"]
    with PlanValidator(name="klimax-val") as validator:
        checked = validator.validate(problem, result.plan)
    assert checked.metric_evaluations == {problem.quality_metrics[0]: 3}


def test_anytime_gets_cheaper():
    problem = roads("costs")
    costs = []
    with AnytimePlanner(name="klimax") as planner, \
            PlanValidator(name="klimax-val") as validator:
        for result in planner.get_solutions(problem, timeout=10):
            checked = validator.validate(problem, result.plan)
            assert checked.status == V.VALID
            costs.append(checked.metric_evaluations[problem.quality_metrics[0]])
    assert result.status == S.SOLVED_OPTIMALLY
    assert costs == sorted(costs, reverse=True) and len(set(costs)) == len(costs)
    assert costs[-1] == 3


def test_the_validator_counts_the_length():
    problem = trip("shortest")
    with OneshotPlanner(name="klimax") as planner:
        plan = planner.solve(problem, timeout=10).plan
    with PlanValidator(name="klimax-val") as validator:
        checked = validator.validate(problem, plan)
    assert checked.metric_evaluations == {problem.quality_metrics[0]: 4}


def test_time():
    for variant, ends in TIMES.items():
        problem = jobs(variant)
        with OneshotPlanner(name="klimax") as planner:
            result = planner.solve(problem, timeout=10)
        assert result.status == S.SOLVED_SATISFICING, variant
        flat = result.plan.action_plan
        assert isinstance(flat, TimeTriggeredPlan), variant
        made = max(start + (duration or 0) for start, _, duration in flat.timed_actions)
        if ends is not None:
            assert made == ends, variant
        with PlanValidator(name="klimax-val") as validator:
            assert validator.validate(problem, result.plan).status == V.VALID, variant
    # Nobody starts before the literal that wakes them up
    with OneshotPlanner(name="klimax") as planner:
        plan = planner.solve(jobs("timed"), timeout=10).plan
    assert all(start > 7 for start, _, _ in plan.action_plan.timed_actions)


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

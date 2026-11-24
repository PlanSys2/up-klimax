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

"""Klimax from UPF: plans with their decomposition, statuses, its validator."""

import pathlib

import pytest
from unified_planning.engines import PlanGenerationResultStatus as S
from unified_planning.engines import ValidationResultStatus as V
from unified_planning.io import PDDLReader
from unified_planning.model.htn import HierarchicalProblem, Method
from unified_planning.plans import ActionInstance, PlanKind, SequentialPlan
from unified_planning.plans.hierarchical_plan import Decomposition, HierarchicalPlan
from unified_planning.shortcuts import (
    AnytimePlanner, BoolType, Equals, Fluent, InstantaneousAction,
    MinimizeSequentialPlanLength, Not, Object, OneshotPlanner, Or, PlanValidator, UserType,
    get_environment)
from unified_planning.test.examples import get_example_problems

import up_klimax

up_klimax.register()
get_environment().credits_stream = None

HOUSE = pathlib.Path(__file__).parent / "data" / "house"


def solve(problem, **kwargs):
    with OneshotPlanner(name="klimax") as planner:
        return planner.solve(problem, **kwargs)


def validate(problem, plan):
    with PlanValidator(name="klimax-val") as validator:
        return validator.validate(problem, plan)


def delivery(rooms=4, partial=False, wings=True):
    """A robot that carries parcels along a corridor: `deliver` takes one to a room; `go` moves
    the robot there, a step at a time."""
    Room, Parcel = UserType("Room"), UserType("Parcel")
    at = Fluent("at", BoolType(), r=Room)
    parcel_in = Fluent("in", BoolType(), p=Parcel, r=Room)
    holding = Fluent("holding", BoolType(), p=Parcel)
    link = Fluent("link", BoolType(), a=Room, b=Room)
    p = HierarchicalProblem("delivery")
    for f in (at, parcel_in, holding, link):
        p.add_fluent(f, default_initial_value=False)
    move = InstantaneousAction("move", a=Room, b=Room)
    move.add_precondition(at(move.a))
    move.add_precondition(link(move.a, move.b))
    move.add_effect(at(move.a), False)
    move.add_effect(at(move.b), True)
    pick = InstantaneousAction("pick", x=Parcel, r=Room)
    pick.add_precondition(at(pick.r))
    pick.add_precondition(parcel_in(pick.x, pick.r))
    pick.add_effect(parcel_in(pick.x, pick.r), False)
    pick.add_effect(holding(pick.x), True)
    drop = InstantaneousAction("drop", x=Parcel, r=Room)
    drop.add_precondition(at(drop.r))
    drop.add_precondition(holding(drop.x))
    drop.add_effect(holding(drop.x), False)
    drop.add_effect(parcel_in(drop.x, drop.r), True)
    p.add_actions([move, pick, drop])
    go = p.add_task("go", to=Room)
    deliver = p.add_task("deliver", x=Parcel, to=Room)
    here = Method("go-here", to=Room)
    here.set_task(go, here.to)
    here.add_precondition(at(here.to))
    p.add_method(here)
    step = Method("go-step", a=Room, b=Room, to=Room)
    step.set_task(go, step.to)
    step.add_precondition(at(step.a))
    first = step.add_subtask(move, step.a, step.b)
    rest = step.add_subtask(go, step.to)
    step.set_ordered(first, rest)
    p.add_method(step)
    carry = Method("carry", x=Parcel, src=Room, to=Room)
    carry.set_task(deliver, carry.x, carry.to)
    carry.add_precondition(parcel_in(carry.x, carry.src))
    s1 = carry.add_subtask(go, carry.src)
    s2 = carry.add_subtask(pick, carry.x, carry.src)
    s3 = carry.add_subtask(go, carry.to)
    s4 = carry.add_subtask(drop, carry.x, carry.to)
    carry.set_ordered(s1, s2, s3, s4)
    p.add_method(carry)
    rs = [Object(f"r{i}", Room) for i in range(rooms)]
    box, book = Object("box", Parcel), Object("book", Parcel)
    p.add_objects(rs + [box, book])
    p.set_initial_value(at(rs[0]), True)
    for a, b in zip(rs, rs[1:]):
        p.set_initial_value(link(a, b), True)
        if wings:
            p.set_initial_value(link(b, a), True)
    p.set_initial_value(parcel_in(box, rs[-1]), True)
    p.set_initial_value(parcel_in(book, rs[1]), True)
    t1 = p.task_network.add_subtask(deliver, box, rs[0])
    t2 = p.task_network.add_subtask(deliver, book, rs[-1])
    if not partial:
        p.task_network.set_ordered(t1, t2)
    return p, (t1, t2)


def test_the_example_of_upf():
    case = get_example_problems()["htn-go"]
    result = solve(case.problem, timeout=10)
    assert result.status == S.SOLVED_SATISFICING
    assert result.plan.kind == PlanKind.HIERARCHICAL_PLAN
    assert validate(case.problem, result.plan).status == V.VALID
    for plan in case.valid_plans:
        assert validate(case.problem, plan).status == V.VALID


@pytest.mark.parametrize("partial", [False, True])
def test_plan_and_decomposition(partial):
    problem, (t1, t2) = delivery(partial=partial)
    result = solve(problem, timeout=10)
    assert result.status == S.SOLVED_SATISFICING
    plan = result.plan
    assert set(plan.decomposition.subtasks) == {t1.identifier, t2.identifier}
    top = plan.decomposition.subtasks[t1.identifier]
    assert top.method == problem.method("carry")
    assert [str(v) for v in top.parameters] == ["box", "r3", "r0"]
    # The actions of the decomposition are those of the sequence, the same objects
    in_tree = [a for _, a in plan.actions()]
    assert sorted(map(id, in_tree)) == sorted(map(id, plan.action_plan.actions))
    assert all(a.action in problem.actions for a in in_tree)
    assert validate(problem, plan).status == V.VALID
    assert float(result.metrics["engine_internal_time"]) >= 0


def test_unsolvable_is_proven():
    problem, _ = delivery(wings=False)  # the corridor only goes one way: no coming back
    assert solve(problem, timeout=10).status == S.UNSOLVABLE_PROVEN


def test_invalid_plans():
    problem, (t1, t2) = delivery()
    plan = solve(problem).plan
    actions = plan.action_plan.actions
    # The same decomposition over the actions in another order
    swapped = HierarchicalPlan(
        SequentialPlan([actions[1], actions[0]] + actions[2:]), plan.decomposition)
    assert validate(problem, swapped).status == V.INVALID
    # A task of the initial network decomposed as the other one
    crossed = HierarchicalPlan(plan.action_plan, Decomposition({
        t1.identifier: plan.decomposition.subtasks[t2.identifier],
        t2.identifier: plan.decomposition.subtasks[t1.identifier]}))
    assert validate(problem, crossed).status == V.INVALID


def test_constraints_between_parameters():
    """What HDDL's :constraints cannot say goes to the precondition: here, a disjunction and
    an inequality between the parameters of a method."""
    problem, _ = delivery()
    rooms = [problem.object(f"r{i}") for i in range(4)]
    go, move = problem.get_task("go"), problem.action("move")
    visit = problem.add_task("visit")
    m = Method("visit-an-end", a=UserType("Room"), b=UserType("Room"))
    m.set_task(visit)
    m.add_constraint(Or(Equals(m.a, rooms[0]), Equals(m.a, rooms[3])))
    m.add_constraint(Not(Equals(m.a, m.b)))
    m.add_constraint(Equals(m.b, rooms[2]))
    s1 = m.add_subtask(go, m.a)
    s2 = m.add_subtask(go, m.b)
    m.set_ordered(s1, s2)
    problem.add_method(m)
    last = problem.task_network.add_subtask(visit)
    result = solve(problem, timeout=10)
    assert result.status == S.SOLVED_SATISFICING
    chosen = result.plan.decomposition.subtasks[last.identifier]
    assert str(chosen.parameters[0]) in ("r0", "r3") and str(chosen.parameters[1]) == "r2"
    assert validate(problem, result.plan).status == V.VALID
    assert move in problem.actions


def test_hddl_read_by_upf():
    reader = PDDLReader()
    for name in ("p1-total-order", "p2-partial-order"):
        problem = reader.parse_problem(str(HOUSE / "domain.hddl"), str(HOUSE / f"{name}.hddl"))
        result = solve(problem, timeout=10)
        assert result.status == S.SOLVED_SATISFICING, name
        assert validate(problem, result.plan).status == V.VALID, name


def test_what_it_does_not_plan():
    case = get_example_problems()["htn-go-temporal"]
    assert not up_klimax.KlimaxEngine.supports(case.problem.kind)
    assert not up_klimax.KlimaxEngine.supports(get_example_problems()["basic"].problem.kind)


def test_chosen_by_upf_for_a_hierarchical_problem():
    problem, _ = delivery()
    with OneshotPlanner(problem_kind=problem.kind) as planner:
        assert planner.name in ("klimax", "aries")  # whichever is installed first


def test_options():
    problem, _ = delivery()
    with OneshotPlanner(name="klimax", params={"timeout": 5}) as planner:
        assert planner.solve(problem).status == S.SOLVED_SATISFICING
    with pytest.raises(Exception):
        OneshotPlanner(name="klimax", params={"threads": 2}).__enter__()


def detour():
    """`delivery` with another way of going somewhere, which stops to rest first: plans with
    it are longer."""
    problem, tasks = delivery()
    Room = problem.user_type("Room")
    go = problem.get_task("go")
    rest = InstantaneousAction("rest")
    problem.add_action(rest)
    slow = Method("go-after-a-rest", to=Room)
    slow.set_task(go, slow.to)
    s1 = slow.add_subtask(rest)
    s2 = slow.add_subtask(go, slow.to)
    slow.set_ordered(s1, s2)
    problem.add_method(slow)
    return problem, tasks


def test_anytime_plans_get_shorter():
    problem, _ = detour()
    problem.add_quality_metric(MinimizeSequentialPlanLength())
    with AnytimePlanner(name="klimax") as planner:
        results = list(planner.get_solutions(problem, timeout=20))
    assert [r.status for r in results[:-1]] == [S.INTERMEDIATE] * (len(results) - 1)
    assert results[-1].status == S.SOLVED_OPTIMALLY
    lengths = [len(r.plan.action_plan.actions) for r in results]
    assert lengths == sorted(lengths, reverse=True) and len(set(lengths[:-1])) == len(lengths) - 1
    assert lengths[-1] == 13
    for r in results:
        assert validate(problem, r.plan).status == V.VALID


def test_the_shortest_plan_when_asked_for():
    problem, _ = detour()
    # Without a metric: any plan, and no claim about it
    assert solve(problem, timeout=20).status == S.SOLVED_SATISFICING
    problem.add_quality_metric(MinimizeSequentialPlanLength())
    result = solve(problem, timeout=20)
    assert result.status == S.SOLVED_OPTIMALLY
    assert len(result.plan.action_plan.actions) == 13
    assert validate(problem, result.plan).status == V.VALID
    # Not asked to go on: the first plan
    with OneshotPlanner(name="klimax", params={"anytime": False}) as planner:
        assert planner.solve(problem, timeout=20).status == S.SOLVED_SATISFICING


def test_anytime_without_a_plan():
    problem, _ = delivery(wings=False)
    with AnytimePlanner(name="klimax") as planner:
        results = list(planner.get_solutions(problem, timeout=10))
    assert [r.status for r in results] == [S.UNSOLVABLE_PROVEN]

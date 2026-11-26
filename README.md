<p align="center">
  <img src="https://raw.githubusercontent.com/PlanSys2/klimax/main/docs/images/klimax.png" alt="Klimax logo" width="300">
</p>

<h1 align="center">up-klimax</h1>

<p align="center">
  <a href="https://github.com/PlanSys2/up-klimax/actions/workflows/ci.yml"><img src="https://github.com/PlanSys2/up-klimax/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/status-in%20development-orange" alt="Status: in development">
</p>

[Klimax](https://github.com/PlanSys2/klimax), an HTN planner for HDDL, as an engine of the
[Unified Planning Framework](https://github.com/aiplan4eu/unified-planning).

```python
from unified_planning.shortcuts import OneshotPlanner, PlanValidator
import up_klimax

up_klimax.register()  # until UPF lists it among its engines

with OneshotPlanner(name="klimax") as planner:
    result = planner.solve(problem, timeout=10)  # a HierarchicalProblem
print(result.plan.action_plan)    # the actions
print(result.plan.decomposition)  # each task, its method and its subtasks

with PlanValidator(name="klimax-val") as validator:
    print(validator.validate(problem, result.plan).status)
```

## What it gives

- **`klimax`**, a oneshot planner for hierarchical problems. The plan is a `HierarchicalPlan`:
  the sequence of actions and, for every task, the method applied, the values of its
  parameters and what each of its subtasks became. Every plan is verified, independently of
  the search, before it is returned. When there is no plan it says so as a proof
  (`UNSOLVABLE_PROVEN`).
- The same engine as an **anytime planner** (`AnytimePlanner(name="klimax")`): after the first
  plan it gives better ones as it finds them: shorter (`MinimizeSequentialPlanLength`, or no
  metric), cheaper (`MinimizeActionCosts`) or with a better value at the end
  (`MinimizeExpressionOnFinalState`, `MaximizeExpressionOnFinalState`). When Klimax proves
  that the plan is the best, the status is `SOLVED_OPTIMALLY`: it does for the shortest, and
  for the cheapest when every action has a cost above zero that does not depend on the state.
  The value of the metric for the plan is in `result.metrics["klimax_metric"]`.
- **`klimax-replanner`** and **`klimax-repairer`**: `Replanner(problem,
  name="klimax-replanner")` keeps the problem, takes changes to it (initial values, goals,
  actions) and plans again; `PlanRepairer(name="klimax-repairer")` gives a plan back as it is if
  it is still valid for the problem, and plans again if not. UPF's own replanner does not take
  hierarchical problems.
- **`klimax-val`**, a validator of hierarchical plans: it checks the decomposition, not only
  that the actions can be executed, and gives what the plan is worth by the metric of the
  problem (`metric_evaluations`).

- **`klimax-htn-usertype-fluents-remover`**, a compiler for any HTN engine: a hierarchical
  problem with fluents whose value is an object, as one with predicates only. UPF's own
  compiler for it does not take problems with tasks. `up_klimax.plan_back(problem, plan)`
  takes a plan of the result back to the problem, methods included.

## What it plans with

Tasks and methods with preconditions; totally and partially ordered task networks; variables
in the initial task network; constraints between the parameters of a method or of the initial
network; negative, disjunctive, existential and universal conditions; equalities; conditional
and universal effects; flat and hierarchical typing; fluents whose value is an object; fluents
whose value is a number (integer or real), compared in conditions and assigned, increased or
decreased in effects; as quality metrics, the length of the plan, action costs (constants,
static fluents or fluents of the state, integer or real) and the value of an expression at
the end; durative actions (with durations that are constants, fluents or intervals) and timed
effects, for which the plan is a `TimeTriggeredPlan` inside the `HierarchicalPlan`: actions
that neither the hierarchy nor what they read and write order run at once. Each one is tested
(`tests/test_kinds.py`).

Not yet: orderings between the starts and the ends of tasks (`TASK_ORDER_TEMPORAL`: Klimax
reads them in HDDL, but UPF does not write them), problems that need actions to overlap,
numbers with bounds and oversubscription.

HDDL has no words for some of these. They are rewritten before Klimax reads the problem,
keeping every name, so that the plan is a plan of the problem as given
(`up_klimax/convert.py`): fluents with object values become predicates, and constraints that
are not `=` or its negation become part of a precondition.

## Install

From source, with Klimax's Python module built (`klimax-planner`) and on the path:

```
pip install unified-planning
pip install -e .
pytest
```

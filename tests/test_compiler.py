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

"""The compiler: a hierarchical problem without fluents of object value, for any HTN engine."""

from unified_planning.engines import CompilationKind
from unified_planning.engines import PlanGenerationResultStatus as S
from unified_planning.engines import ValidationResultStatus as V
from unified_planning.shortcuts import Compiler, OneshotPlanner, PlanValidator, get_environment
from unified_planning.test.examples import get_example_problems

import up_klimax

up_klimax.register()
get_environment().credits_stream = None


def test_object_fluents_become_predicates():
    case = get_example_problems()["htn-go"]
    problem = case.problem
    with Compiler(name="klimax-htn-usertype-fluents-remover",
                  compilation_kind=CompilationKind.USERTYPE_FLUENTS_REMOVING) as compiler:
        result = compiler.compile(problem)
        expected = compiler.resulting_problem_kind(problem.kind)
    compiled = result.problem
    assert problem.kind.has_object_fluents() and not compiled.kind.has_object_fluents()
    assert compiled.kind.features <= expected.features
    # The same tasks, methods and initial network, constraints included
    assert [t.name for t in compiled.tasks] == [t.name for t in problem.tasks]
    assert [m.name for m in compiled.methods] == [m.name for m in problem.methods]
    assert len(compiled.task_network.constraints) == len(problem.task_network.constraints)
    # A plan of the compiled problem is, taken back, a plan of the problem
    with OneshotPlanner(name="klimax") as planner:
        solved = planner.solve(compiled, timeout=10)
    assert solved.status == S.SOLVED_SATISFICING
    back = up_klimax.plan_back(problem, solved.plan)
    assert all(a.action in problem.actions for _, a in back.actions())
    assert all(m.method in problem.methods for _, m in back.methods())
    with PlanValidator(name="klimax-val") as validator:
        assert validator.validate(problem, back).status == V.VALID
    first = solved.plan.action_plan.actions[0]
    assert result.map_back_action_instance(first).action == problem.action(first.action.name)


def test_chosen_by_upf():
    problem = get_example_problems()["htn-go"].problem
    with Compiler(problem_kind=problem.kind,
                  compilation_kind=CompilationKind.USERTYPE_FLUENTS_REMOVING) as compiler:
        assert compiler.name == "klimax-htn-usertype-fluents-remover"

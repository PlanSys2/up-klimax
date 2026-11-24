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

"""A UPF compiler for hierarchical problems: fluents whose value is an object become
predicates. UPF's own compiler for it only takes problems without tasks."""

from typing import Optional

import unified_planning as up
from unified_planning.engines import CompilationKind, Engine
from unified_planning.engines.mixins.compiler import CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import ProblemKind
from unified_planning.plans import ActionInstance
from unified_planning.plans.hierarchical_plan import (
    Decomposition,
    HierarchicalPlan,
    MethodInstance,
)

from .convert import convert
from .kind import SUPPORTED_KIND


class HtnUsertypeFluentsRemover(Engine, CompilerMixin):
    """``Compiler(name="klimax-htn-usertype-fluents-remover")``.

    The result keeps every name. `map_back_action_instance` gives the actions of the problem;
    for a whole hierarchical plan, methods included, use `plan_back`.
    """

    def __init__(self, **options):
        Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.USERTYPE_FLUENTS_REMOVING)
        if options:
            raise up.exceptions.UPUsageError(f"unknown options: {sorted(options)}")

    @property
    def name(self) -> str:
        return "klimax-htn-usertype-fluents-remover"

    @staticmethod
    def supported_kind() -> ProblemKind:
        return SUPPORTED_KIND

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind.has_hierarchical() and problem_kind <= SUPPORTED_KIND

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.USERTYPE_FLUENTS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        if new_kind.has_object_fluents():
            new_kind.unset_fluents_type("OBJECT_FLUENTS")
            new_kind.set_effects_kind("CONDITIONAL_EFFECTS")
            new_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
            new_kind.set_conditions_kind("EQUALITIES")
            new_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        return new_kind

    def _compile(self, problem, compilation_kind) -> CompilerResult:
        compiled = convert(problem, for_hddl=False).problem
        compiled.name = f"{self.name}_{problem.name}"

        def action_back(instance):
            return ActionInstance(problem.action(instance.action.name),
                                  instance.actual_parameters)
        return CompilerResult(compiled, action_back, self.name)


def plan_back(problem, plan) -> HierarchicalPlan:
    """A hierarchical plan of a problem compiled from `problem` (same names), as a plan of
    `problem`: its actions and its methods."""
    actions = {}

    def back(instance):
        if isinstance(instance, ActionInstance):
            if id(instance) not in actions:
                actions[id(instance)] = ActionInstance(
                    problem.action(instance.action.name), instance.actual_parameters)
            return actions[id(instance)]
        return MethodInstance(problem.method(instance.method.name), instance.parameters,
                              tree(instance.decomposition))

    def tree(decomposition):
        return Decomposition({label: back(i) for label, i in decomposition.subtasks.items()})

    root = tree(plan.decomposition)
    flat = up.plans.SequentialPlan(
        [back(a) for a in plan.action_plan.actions], problem.environment)
    return HierarchicalPlan(flat, root)

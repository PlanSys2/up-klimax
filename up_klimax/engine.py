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

"""Klimax as an engine of the Unified Planning Framework: planning in the same process
(klimax-planner) for hierarchical problems, and its verifier of hierarchical plans."""

from typing import Optional

import klimax
import unified_planning as up
from unified_planning.engines import Credits, Engine
from unified_planning.engines.mixins.oneshot_planner import (
    OneshotPlannerMixin,
    OptimalityGuarantee,
)
from unified_planning.engines.mixins.plan_validator import PlanValidatorMixin
from unified_planning.engines.results import (
    LogLevel,
    LogMessage,
    PlanGenerationResult,
    PlanGenerationResultStatus,
    ValidationResult,
    ValidationResultStatus,
)
from unified_planning.io import PDDLWriter
from unified_planning.plans import ActionInstance, SequentialPlan
from unified_planning.plans.hierarchical_plan import (
    Decomposition,
    HierarchicalPlan,
    MethodInstance,
)

from .convert import TOP_METHOD, convert, order
from .kind import SUPPORTED_KIND

credits = Credits(
    "Klimax",
    "Intelligent Robotics Lab, Universidad Rey Juan Carlos",
    "fmrico@gmail.com",
    "https://github.com/PlanSys2/klimax",
    "Apache-2.0",
    "HTN planner for HDDL, with total and partial order, and proofs of unsolvability.",
    "Klimax plans hierarchical problems (tasks, methods with preconditions and constraints, "
    "totally and partially ordered task networks) and proves when no plan exists. Every plan "
    "is verified, independently of the search, before it is returned.",
)

_NO_PLAN = {
    klimax.Status.UNSOLVABLE: PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    klimax.Status.TIMEOUT: PlanGenerationResultStatus.TIMEOUT,
    klimax.Status.UNSUPPORTED: PlanGenerationResultStatus.UNSUPPORTED_PROBLEM,
    klimax.Status.INVALID_INPUT: PlanGenerationResultStatus.INTERNAL_ERROR,
    klimax.Status.INTERNAL_ERROR: PlanGenerationResultStatus.INTERNAL_ERROR,
}

_LOG_LEVEL = {
    klimax.Severity.ERROR: LogLevel.ERROR,
    klimax.Severity.WARNING: LogLevel.WARNING,
    klimax.Severity.NOTE: LogLevel.INFO,
}


def _hddl(problem):
    """The problem as Klimax reads it: how it was rewritten, the writer (which maps the names
    back) and the two texts."""
    converted = convert(problem)
    writer = PDDLWriter(converted.problem)
    return converted, writer, writer.get_domain(), writer.get_problem()


def _logs(diagnostics):
    return [LogMessage(_LOG_LEVEL[d.severity], str(d)) for d in diagnostics]


class KlimaxEngine(Engine, OneshotPlannerMixin):
    """Klimax in UPF (its verifier is KlimaxValidator, "klimax-val").

    Options (``OneshotPlanner(name="klimax", params={...})``):

    - ``timeout``: seconds when the call gives none (default 30).
    """

    def __init__(self, **options):
        Engine.__init__(self)
        OneshotPlannerMixin.__init__(self)
        unknown = set(options) - {"timeout"}
        if unknown:
            raise up.exceptions.UPUsageError(f"unknown Klimax options: {sorted(unknown)}")
        self._options = options

    @property
    def name(self) -> str:
        return "klimax"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        return SUPPORTED_KIND

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return problem_kind.has_hierarchical() and problem_kind <= SUPPORTED_KIND

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        return optimality_guarantee == OptimalityGuarantee.SATISFICING

    @staticmethod
    def get_credits(**kwargs) -> Optional[Credits]:
        return credits

    def _plan_back(self, problem, converted, writer, result):
        """Klimax's plan as a UPF plan of `problem` (names as the writer gave them)."""
        em = problem.environment.expression_manager
        steps = {s.id: s for s in result.steps_of_plan}
        built = {}

        def objects(names):
            return [em.ObjectExp(writer.get_item_named(n)) for n in names]

        def decomposition(network, ids):
            return Decomposition(
                {label: instance(i) for label, i in zip(order(network), ids, strict=True)})

        def instance(step_id):
            if step_id in built:
                return built[step_id]
            step = steps[step_id]
            if step.primitive:
                action = problem.action(writer.get_item_named(step.task).name)
                made = ActionInstance(action, objects(step.args))
            else:
                method = writer.get_item_named(step.method)
                made = MethodInstance(problem.method(method.name), tuple(objects(step.params)),
                                      decomposition(method, step.subtasks))
            built[step_id] = made
            return made

        if converted.top:
            top = steps[result.roots[0]]
            root = decomposition(converted.problem.method(TOP_METHOD), top.subtasks)
        else:
            root = decomposition(converted.problem.task_network, result.roots)
        flat = SequentialPlan([instance(i) for i in result.sequence], problem.environment)
        return HierarchicalPlan(flat, root)

    def _solve(self, problem, heuristic=None, timeout=None, output_stream=None):
        if heuristic is not None:
            import warnings
            warnings.warn("Klimax does not use a custom heuristic", stacklevel=3)
        try:
            converted, writer, domain_text, problem_text = _hddl(problem)
        except (up.exceptions.UPException, ValueError) as e:
            return PlanGenerationResult(
                PlanGenerationResultStatus.UNSUPPORTED_PROBLEM, None, self.name,
                log_messages=[LogMessage(LogLevel.ERROR, str(e))])
        seconds = float(timeout if timeout is not None else self._options.get("timeout", 30))
        result = klimax.solve(domain_text, problem_text, timeout=seconds)
        logs = _logs(result.diagnostics)
        if result.message:
            logs.append(LogMessage(LogLevel.INFO, result.message))
        if output_stream is not None:
            output_stream.write(f"[klimax] {result.status.name.lower()}: ground "
                                f"{result.ground_ms:.1f} ms, search {result.search_ms:.1f} ms, "
                                f"{result.expansions} expansions\n")
        metrics = {
            "engine_internal_time": str((result.ground_ms + result.search_ms) / 1000.0),
            "klimax_ground_ms": str(result.ground_ms),
            "klimax_search_ms": str(result.search_ms),
            "klimax_expansions": str(result.expansions),
        }
        if result.status != klimax.Status.SOLVED:
            return PlanGenerationResult(_NO_PLAN[result.status], None, self.name, metrics, logs)
        plan = self._plan_back(problem, converted, writer, result)
        return PlanGenerationResult(
            PlanGenerationResultStatus.SOLVED_SATISFICING, plan, self.name, metrics, logs)


class KlimaxValidator(Engine, PlanValidatorMixin):
    """Klimax's verifier in UPF ("klimax-val"): hierarchical plans, with the decomposition of
    every task."""

    def __init__(self, **options):
        Engine.__init__(self)
        PlanValidatorMixin.__init__(self)
        if options:
            raise up.exceptions.UPUsageError(f"unknown Klimax options: {sorted(options)}")

    @property
    def name(self) -> str:
        return "klimax-val"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        return SUPPORTED_KIND

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return problem_kind.has_hierarchical() and problem_kind <= SUPPORTED_KIND

    @staticmethod
    def supports_plan(plan_kind: "up.plans.PlanKind") -> bool:
        return plan_kind == up.plans.PlanKind.HIERARCHICAL_PLAN

    @staticmethod
    def get_credits(**kwargs) -> Optional[Credits]:
        return credits

    def _validate(self, problem, plan) -> ValidationResult:
        try:
            converted, writer, domain_text, problem_text = _hddl(problem)
            text = _plan_text(converted, writer, plan)
        except (up.exceptions.UPException, ValueError, KeyError) as e:
            return ValidationResult(ValidationResultStatus.UNKNOWN, self.name,
                                    [LogMessage(LogLevel.ERROR, str(e))])
        report = klimax.verify(domain_text, problem_text, text)
        logs = _logs(report.diagnostics)
        if report.valid:
            return ValidationResult(ValidationResultStatus.VALID, self.name, logs)
        logs.append(LogMessage(LogLevel.INFO, report.reason))
        unknown = bool(report.diagnostics) or report.reason.startswith("unsupported")
        return ValidationResult(
            ValidationResultStatus.UNKNOWN if unknown else ValidationResultStatus.INVALID,
            self.name, logs)


def _linear(network):
    """The identifiers of the subtasks of a network in an order it allows: the IPC format
    wants the subtasks of a step written so."""
    total = network.total_order()
    if total is not None:
        return list(total)
    before = network.partial_order()
    if before is None:
        raise ValueError("a task network with orderings HDDL cannot express")
    pending = [s.identifier for s in network.subtasks]
    out = []
    while pending:
        free = next(i for i in pending
                    if not any(b == i and a in pending for a, b in before))
        pending.remove(free)
        out.append(free)
    return out


def _plan_text(converted, writer, plan) -> str:
    """A UPF hierarchical plan in the format of the IPC HTN tracks, for the problem as Klimax
    reads it."""
    problem = converted.problem
    name = writer.get_pddl_name
    ids = {}  # of each action instance, by identity
    lines = ["==>"]
    for n, a in enumerate(plan.action_plan.actions):
        ids[id(a)] = n + 1
        args = " ".join(name(p.object()) for p in a.actual_parameters)
        lines.append(f"{n + 1} {name(problem.action(a.action.name))} {args}".rstrip())
    tree = []
    counter = [len(ids)]

    def children(network, decomposition):
        return " ".join(str(step(decomposition.subtasks[label])) for label in _linear(network))

    def step(instance) -> int:
        if isinstance(instance, ActionInstance):
            return ids[id(instance)]
        counter[0] += 1
        mine = counter[0]
        method = problem.method(instance.method.name)
        value = {p.name: name(v.object())
                 for p, v in zip(method.parameters, instance.parameters, strict=True)}
        task = method.achieved_task
        args = " ".join(value[p.name] for p in task.parameters)
        tree.append(f"{mine} {name(task.task)} {args}".rstrip() + f" -> {name(method)} " +
                    children(method, instance.decomposition))
        return mine

    if converted.top:
        counter[0] += 1
        top = counter[0]
        method = problem.method(TOP_METHOD)
        tree.append(f"{top} {name(method.achieved_task.task)} -> {name(method)} " +
                    children(method, plan.decomposition))
        lines.append(f"root {top}")
    else:
        lines.append("root " + children(problem.task_network, plan.decomposition))
    return "\n".join(lines + tree + ["<=="]) + "\n"

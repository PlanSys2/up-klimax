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

"""UPF hierarchical problems as Klimax reads them: what HDDL has no words for is rewritten
first, keeping every name, so that a plan of the result is a plan of the problem.

- Fluents whose value is an object become predicates with one more parameter (UPF's own
  rewriting, which it only applies to problems without tasks).
- Constraints between the parameters of a method that are not orderings (anything but = and
  its negation has no place in HDDL's :constraints) become part of its precondition: they do not
  depend on the state.
- Those of the initial task network, which has no precondition, go to a method of a new task
  that stands for the whole network. A plan leaves that task out.
"""

from collections import OrderedDict

from unified_planning.engines import CompilationKind
from unified_planning.engines.compilers.usertype_fluents_remover import UsertypeFluentsRemover
from unified_planning.model.htn import HierarchicalProblem, Method, Subtask, Task
from unified_planning.model.walkers.usertype_fluents_walker import UsertypeFluentsWalker

TOP = "klimax-top"  # the task that stands for the initial network
TOP_METHOD = "klimax-top-method"


class Converted:
    """`problem` rewritten; `top` if its initial network is the subtasks of method TOP."""

    def __init__(self, problem, top):
        self.problem = problem
        self.top = top


def convert(problem) -> Converted:
    env = problem.environment
    em = env.expression_manager
    flat = problem
    condition = lambda e: e  # noqa: E731
    if problem.kind.has_object_fluents():
        flat = UsertypeFluentsRemover()._compile(
            problem, CompilationKind.USERTYPE_FLUENTS_REMOVING).problem
        fluents = {f: flat.fluent(f.name) for f in problem.fluents if f.type.is_user_type()}
        names = {f.name for f in problem.fluents} | {a.name for a in problem.actions} | \
            {o.name for o in problem.all_objects}
        condition = UsertypeFluentsWalker(
            fluents, names, env).remove_usertype_fluents_from_condition

    out = HierarchicalProblem(problem.name, env)
    for f in flat.fluents:
        out.add_fluent(f, default_initial_value=flat.fluents_defaults.get(f))
    out.add_objects(flat.all_objects)
    for a in flat.actions:
        out.add_action(a)
    for f, v in flat.explicit_initial_values.items():
        out.set_initial_value(f, v)
    for g in flat.goals:
        out.add_goal(g)
    for t in problem.tasks:
        out.add_task(t)

    def network(source, target, substitution, precondition):
        """The subtasks and orderings of `source` in `target`; what else it constrains goes
        to `precondition`."""
        for s in source.subtasks:
            task = out.action(s.task.name) if not isinstance(s.task, Task) else s.task
            target.add_subtask(Subtask(
                task, *[p.substitute(substitution) for p in s.parameters], ident=s.identifier))
        for c in source.temporal_constraints():
            target.add_constraint(c)
        for c in source.non_temporal_constraints():
            precondition(condition(c).substitute(substitution))

    for m in problem.methods:
        new = Method(m.name, OrderedDict((p.name, p.type) for p in m.parameters), env)
        to_new = {em.ParameterExp(p): em.ParameterExp(new.parameter(p.name))
                  for p in m.parameters}
        new.set_task(m.achieved_task.task,
                     *[new.parameter(p.name) for p in m.achieved_task.parameters])
        for p in m.preconditions:
            new.add_precondition(condition(p).substitute(to_new))
        network(m, new, to_new, new.add_precondition)
        out.add_method(new)

    tn = problem.task_network
    top = bool(tn.non_temporal_constraints())
    if not top:
        to_new = {em.ParameterExp(v): em.ParameterExp(out.task_network.add_variable(
            v.name, v.type)) for v in tn.variables}
        network(tn, out.task_network, to_new, None)
    else:
        if out.has_name(TOP) or out.has_name(TOP_METHOD):
            raise ValueError(f"the name {TOP} is reserved")
        task = out.add_task(TOP)
        method = Method(TOP_METHOD, OrderedDict((v.name, v.type) for v in tn.variables), env)
        method.set_task(task)
        to_new = {em.ParameterExp(v): em.ParameterExp(method.parameter(v.name))
                  for v in tn.variables}
        network(tn, method, to_new, method.add_precondition)
        out.add_method(method)
        out.task_network.add_subtask(Subtask(task, ident=TOP))
    return Converted(out, top)


def order(network):
    """The identifiers of the subtasks of a network in the order UPF's writer writes them:
    Klimax gives the subtasks of a step in that order."""
    total = network.total_order()
    return list(total) if total is not None else [s.identifier for s in network.subtasks]

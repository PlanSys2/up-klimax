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

"""What Klimax plans with, as UPF problem kinds (each one tested: tests/test_kinds.py)."""

from unified_planning.model import ProblemKind
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION

FEATURES = {
    "PROBLEM_CLASS": ["HIERARCHICAL"],
    # Not TASK_ORDER_TEMPORAL: no time yet
    "HIERARCHICAL": ["METHOD_PRECONDITIONS", "TASK_NETWORK_CONSTRAINTS",
                     "INITIAL_TASK_NETWORK_VARIABLES", "TASK_ORDER_TOTAL",
                     "TASK_ORDER_PARTIAL"],
    "CONDITIONS_KIND": ["NEGATIVE_CONDITIONS", "DISJUNCTIVE_CONDITIONS", "EQUALITIES",
                        "EXISTENTIAL_CONDITIONS", "UNIVERSAL_CONDITIONS"],
    "EFFECTS_KIND": ["CONDITIONAL_EFFECTS", "FORALL_EFFECTS", "INCREASE_EFFECTS",
                     "DECREASE_EFFECTS", "STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS",
                     "FLUENTS_IN_NUMERIC_ASSIGNMENTS"],
    "TYPING": ["FLAT_TYPING", "HIERARCHICAL_TYPING"],
    # Object fluents as predicates (convert.py). Numbers without bounds: BOUNDED_TYPES is not
    # declared, since nothing in HDDL says that a value must stay within limits
    "FLUENTS_TYPE": ["OBJECT_FLUENTS", "INT_FLUENTS", "REAL_FLUENTS"],
    "PROBLEM_TYPE": ["SIMPLE_NUMERIC_PLANNING", "GENERAL_NUMERIC_PLANNING"],
    "QUALITY_METRICS": ["PLAN_LENGTH"],
}


def _kind() -> ProblemKind:
    kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
    for group, features in FEATURES.items():
        for f in features:
            getattr(kind, "set_" + group.lower())(f)
    return kind


SUPPORTED_KIND = _kind()

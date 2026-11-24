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

"""Klimax as an engine of the Unified Planning Framework.

    import up_klimax
    up_klimax.register()            # until UPF lists it among its engines
    with OneshotPlanner(name="klimax") as planner:
        result = planner.solve(hierarchical_problem, timeout=10)

Also "klimax-val", a validator of hierarchical plans; "klimax-replanner" and "klimax-repairer";
and a compiler that other HTN engines can
use: "klimax-htn-usertype-fluents-remover" (fluents with object values as predicates).
"""

from .compiler import HtnUsertypeFluentsRemover, plan_back
from .engine import KlimaxEngine, KlimaxValidator
from .replan import KlimaxRepairer, KlimaxReplanner

__version__ = "0.0.1"
__all__ = ["HtnUsertypeFluentsRemover", "KlimaxEngine", "KlimaxRepairer", "KlimaxReplanner",
           "KlimaxValidator", "plan_back", "register"]


def register(environment=None):
    """Adds Klimax to the engines of the UPF environment (the global one by default)."""
    from unified_planning.environment import get_environment
    env = environment if environment is not None else get_environment()
    if "klimax" not in env.factory.engines:
        env.factory.add_engine("klimax", __name__, "KlimaxEngine")
    if "klimax-val" not in env.factory.engines:
        env.factory.add_engine("klimax-val", __name__, "KlimaxValidator")
    if "klimax-replanner" not in env.factory.engines:
        env.factory.add_engine("klimax-replanner", __name__, "KlimaxReplanner")
    if "klimax-repairer" not in env.factory.engines:
        env.factory.add_engine("klimax-repairer", __name__, "KlimaxRepairer")
    if "klimax-htn-usertype-fluents-remover" not in env.factory.engines:
        env.factory.add_engine(
            "klimax-htn-usertype-fluents-remover", __name__, "HtnUsertypeFluentsRemover")

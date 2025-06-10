

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


# --------------- Custom Robot Config---------------------------------
from .base.legged_robot import LeggedRobot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot


from legged_gym.envs.g1.g1_full_config import G1Cfg, G1CfgPPO
from legged_gym.envs.g1.g1_full import G1
# --------------------------------------------------------------------



# from .g1.g1 import G1
# from .g1.g1_config import G1Cfg, G1CfgPPO
# from .g1.g1_dof12 import G1Dof12
# from .g1.g1_dof12_config import G1Dof12Cfg, G1Dof12CfgPPO

import os

from legged_gym.utils.task_registry import task_registry

# --------------- Custom Task registry---------------------------------
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "g1_full", G1, G1Cfg(), G1CfgPPO())
# --------------------------------------------------------------------

# task_registry.register( "g1", G1, G1Cfg(), G1CfgPPO() )
# task_registry.register( "g1_dof12", G1Dof12, G1Dof12Cfg(), G1Dof12CfgPPO())
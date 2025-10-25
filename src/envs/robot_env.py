import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
import os

from isaaclab_assets import G1_MINIMAL_CFG

##
# Scene definition
##

@configclass
class BalanceSceneCfg(InteractiveSceneCfg):
    """Scene with G1 robot and skateboard."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # G1 robot (using pre-built Isaac Lab asset)
    robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Skateboard (from your URDF with meshes - as articulation to keep all parts)
    skateboard: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Skateboard",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.join(os.path.dirname(__file__), "..", "assets", "skateboard", "robot.urdf"),
            fix_base=True,  # Fixed to world - won't move
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.08),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["whj.*"],  # Wheel joints
                stiffness=0.0,
                damping=0.1,  # Small damping to keep wheels stable
            ),
            "trucks": ImplicitActuatorCfg(
                joint_names_expr=["trj.*"],  # Truck joints
                stiffness=0.0,
                damping=0.1,
            ),
        },
    )
    
    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action configuration."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation configuration."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward configuration."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    orientation_penalty = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    """Termination configuration."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.3})

@configclass
class EventsCfg:
    """Event configuration."""
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
            "velocity_range": {},
        },
    )

##
# Environment configuration
##

@configclass
class G1BalanceEnvCfg(ManagerBasedRLEnvCfg):
    """G1 robot balancing on skateboard environment."""
    
    scene: BalanceSceneCfg = BalanceSceneCfg(num_envs=4, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    
    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation


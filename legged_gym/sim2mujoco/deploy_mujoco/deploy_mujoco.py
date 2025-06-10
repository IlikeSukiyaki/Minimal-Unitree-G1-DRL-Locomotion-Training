import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


LEGGED_GYM_ROOT_DIR = "/home/yifeng/VScodeproject/g1_gym_terrain/g1_gym"

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    actual_dofs = min(len(target_q), len(q))
    used_kp = kp[:actual_dofs] if len(kp) > actual_dofs else kp
    used_kd = kd[:actual_dofs] if len(kd) > actual_dofs else kd
    
    return (target_q[:actual_dofs] - q[:actual_dofs]) * used_kp + (target_dq[:actual_dofs] - dq[:actual_dofs]) * used_kd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/legged_gym/sim2mujoco/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    actual_dofs = d.qpos.shape[0] - 7
    print(f"Configuration DOFs: {num_actions}, Actual URDF DOFs: {actual_dofs}")
    print(f"Control size: {d.ctrl.shape}")
    
    joint_names = [m.joint(i).name for i in range(m.njnt)]
    print("\nJoint names in the model:")
    for i, name in enumerate(joint_names):
        if 'root' not in name and 'floating' not in name:
            print(f"{i}: {name}")
    
    joint_pos_limits_lower = []
    joint_pos_limits_upper = []
    for i in range(1, m.njnt):
        if i - 1 < actual_dofs:
            try:
                joint_pos_limits_lower.append(m.jnt_range[i][0])
                joint_pos_limits_upper.append(m.jnt_range[i][1])
            except:
                print(f"Warning: No limits found for joint {i}: {joint_names[i]}")
                joint_pos_limits_lower.append(-10.0)
                joint_pos_limits_upper.append(10.0)
    
    joint_pos_limits_lower = np.array(joint_pos_limits_lower, dtype=np.float32)
    joint_pos_limits_upper = np.array(joint_pos_limits_upper, dtype=np.float32)
    
    print("\nJoint position limits from URDF:")
    for i, (lower, upper) in enumerate(zip(joint_pos_limits_lower, joint_pos_limits_upper)):
        joint_name = joint_names[i+1] if i+1 < len(joint_names) else "unknown"
        print(f"{i}: {joint_name}: [{lower:.4f}, {upper:.4f}] rad")
    
    print("\nExpected joint order from policy/config:")
    expected_order = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    for i, name in enumerate(expected_order):
        print(f"{i}: {name}")
    
    joint_map = {}
    for i, expected_name in enumerate(expected_order):
        found = False
        for j, actual_name in enumerate(joint_names):
            if expected_name == actual_name:
                joint_map[i] = j - 1
                found = True
                break
        if not found and i < actual_dofs:
            print(f"WARNING: Joint {expected_name} from policy not found in model")
    
    print("\nJoint mapping (policy_index -> joint_index):")
    for policy_idx, joint_idx in joint_map.items():
        if policy_idx < actual_dofs:
            joint_name = joint_names[joint_idx+1] if joint_idx+1 < len(joint_names) else "INVALID"
            print(f"Policy {policy_idx} ({expected_order[policy_idx]}) -> Joint {joint_idx} ({joint_name})")
    
    has_actuators = d.ctrl.shape[0] > 0
    if not has_actuators:
        print("WARNING: The model does not have any actuators defined. Using qfrc_applied for direct torque control.")

    policy = torch.jit.load(policy_path)

    remapped_target_pos = np.zeros(actual_dofs, dtype=np.float32)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            for policy_idx, joint_idx in joint_map.items():
                if policy_idx < len(target_dof_pos) and joint_idx < actual_dofs:
                    remapped_target_pos[joint_idx] = target_dof_pos[policy_idx]
            
            for i in range(len(remapped_target_pos)):
                if i < len(joint_pos_limits_lower):
                    remapped_target_pos[i] = np.clip(
                        remapped_target_pos[i],
                        joint_pos_limits_lower[i],
                        joint_pos_limits_upper[i]
                    )
                    
                    margin = 0.01
                    if remapped_target_pos[i] > joint_pos_limits_upper[i] - margin:
                        remapped_target_pos[i] = joint_pos_limits_upper[i] - margin
                    elif remapped_target_pos[i] < joint_pos_limits_lower[i] + margin:
                        remapped_target_pos[i] = joint_pos_limits_lower[i] + margin
            
            tau = pd_control(remapped_target_pos, d.qpos[7:], kps[:actual_dofs], np.zeros(actual_dofs), d.qvel[6:], kds[:actual_dofs])
            
            if has_actuators:
                d.ctrl[:] = tau
            else:
                d.qfrc_applied[:] = 0
                d.qfrc_applied[6:6+len(tau)] = tau
            
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                if counter % (control_decimation * 100) == 0:
                    print("\nCurrent joint positions:")
                    for i in range(min(actual_dofs, 12)):
                        joint_idx = i
                        policy_idx = [k for k, v in joint_map.items() if v == joint_idx]
                        policy_idx = policy_idx[0] if policy_idx else -1
                        policy_name = expected_order[policy_idx] if policy_idx >= 0 and policy_idx < len(expected_order) else "UNKNOWN"
                        joint_name = joint_names[joint_idx+1] if joint_idx+1 < len(joint_names) else "UNKNOWN"
                        limit_info = ""
                        if i < len(joint_pos_limits_lower):
                            percent = (d.qpos[joint_idx+7] - joint_pos_limits_lower[i]) / (joint_pos_limits_upper[i] - joint_pos_limits_lower[i]) * 100
                            limit_info = f"Limit: [{joint_pos_limits_lower[i]:.2f}, {joint_pos_limits_upper[i]:.2f}] ({percent:.1f}%)"
                        print(f"Joint {joint_idx} ({joint_name}): {d.qpos[joint_idx+7]:.3f}, Target: {remapped_target_pos[joint_idx]:.3f}, Policy idx: {policy_idx} ({policy_name}), {limit_info}")
                
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                
                if len(qj) < num_actions:
                    qj_padded = np.zeros(num_actions, dtype=np.float32)
                    dqj_padded = np.zeros(num_actions, dtype=np.float32)
                    qj_padded[:len(qj)] = qj
                    dqj_padded[:len(dqj)] = dqj
                    qj = qj_padded
                    dqj = dqj_padded

                qj = (qj - default_angles[:len(qj)]) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                if num_actions == 12:
                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9:21] = qj[:12]
                    obs[21:33] = dqj[:12]
                    obs[33:45] = action
                    obs[45:47] = np.array([sin_phase, cos_phase])
                else:
                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9:9+num_actions] = qj
                    obs[9+num_actions:9+2*num_actions] = dqj
                    obs[9+2*num_actions:9+3*num_actions] = action

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

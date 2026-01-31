"""This script demonstrates position control of the Arm_X4 robot with viser GUI.

.. code-block:: bash

    # Usage
    python control.py

    # Then open http://localhost:8080 in your browser to control the robot.

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Arm_X4 robot position control with viser GUI.")
parser.add_argument(
    "--viser_port",
    type=int,
    default=8080,
    help="Port for viser server.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import viser
from viser.extras import ViserUrdf
import yourdfpy

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCamera, TiledCameraCfg
##
# Pre-defined configs
##
from Arm_X4.robots.arm_x4 import ARM_X4_CFG
from Arm_X4.assets import ARM_X4_URDF_PATH


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for Arm_X4 robot control."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ARM_X4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #Wrist Camera
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0., 0., 0.05),
            rot=(1., 0., 0., 0.),
            convention="world"
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0)
        ),
        width=128,
        height=128,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulation loop."""
    # Extract scene entities
    robot: Articulation = scene["robot"]
    wrist_camera: TiledCamera = scene["wrist_camera"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Print robot info
    print("\n" + "=" * 50)
    print("Robot loaded successfully!")
    print("=" * 50)
    print(f"Number of joints: {robot.num_joints}")
    print(f"Joint names: {robot.joint_names}")
    print(f"Number of bodies: {robot.num_bodies}")
    print(f"Body names: {robot.body_names}")
    print("=" * 50 + "\n")

    # Set camera view
    sim.set_camera_view(eye=np.array([1.5, 1.5, 1.0]), target=np.array([0.0, 0.0, 0.3]))

    # Get device and joint names
    device = robot.device
    joint_names = robot.joint_names

    # Setup viser server
    server = viser.ViserServer(port=args_cli.viser_port)
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)
    print(f"[INFO] Viser server started at http://localhost:{args_cli.viser_port}")

    with server.gui.add_folder("Sensors"):
        gui_rgb = server.gui.add_image(label="Wrist RGB", image=np.zeros((128, 128, 3)), format="jpeg")
        gui_depth = server.gui.add_image(label="Wrist Depth", image=np.zeros((128, 128, 3)), format="jpeg")

    urdf = yourdfpy.URDF.load(ARM_X4_URDF_PATH)
    urdf_vis = ViserUrdf(server, urdf)
    print(f"[INFO] URDF loaded from: {ARM_X4_URDF_PATH}")

    joint_pos_limits = robot.data.joint_pos_limits[0]   # [num_joints, 2]

    sliders = []
    with server.gui.add_folder("Joint Position Control"):
        reset_button = server.gui.add_button("Reset to Zero")

        for joint_name in joint_names:
            i = robot.find_joints(joint_name)[0]
            slider = server.gui.add_slider(
                label=joint_name,
                min=joint_pos_limits[i, 0].item(),
                max=joint_pos_limits[i, 1].item(),
                step=0.01,
                initial_value=0.0,
            )
            sliders.append(slider)

    # Reset button callback
    @reset_button.on_click
    def _(_):
        for slider in sliders:
            slider.value = 0.0

    # Simulation loop
    count = 0
    while simulation_app.is_running():
        slider_values = [slider.value for slider in sliders]
        
        target = torch.tensor([slider_values], dtype=torch.float32, device=device)
        robot.set_joint_position_target(target)

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Update scene
        scene.update(sim_dt)

        # update joint position to viser
        current_joint_pos = robot.data.joint_pos[0].cpu().numpy()
        joint_config = {name: float(pos) for name, pos in zip(joint_names, current_joint_pos)}
        urdf_vis.update_cfg(joint_config)

        # update wrist camera to viser
        rgb_data = wrist_camera.data.output["rgb"][0]
        gui_rgb.image = rgb_data.cpu().numpy().astype(np.uint8)

        depth_data = wrist_camera.data.output["depth"][0]
        depth_data[depth_data == float("inf")] = 0
        gui_depth.image = depth_data.cpu().numpy().astype(np.uint8).repeat(3, axis=2)

        count += 1


def main():
    """Main function."""
    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Setup scene
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Reset simulation
    sim.reset()

    print("[INFO] Setup complete. Running simulation...")

    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

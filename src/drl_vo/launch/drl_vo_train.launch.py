""" 
(Call to velocity_smoother.launch.py, cmd_vel_pub.py and drl_vo_train.py)

This file sets up the training environment for the DRL-VO model. It includes the 
following components:
 - The model file and log directory are specified as launch arguments.
 - The `turtlebot_teleop' pacakage is included for velocity smoothing during teleoperation.
 - The `drl_vo_cmd` node is the main training node, which tuns the training scripts.
 - The `mix_cmd_vel` node handles publishing mixed velocity commands to the robot.

Future modifications:
    - If you plan to modify the robot platform or the simulation environment, focus on the
    `model_file`, and `log_dir`, and any included files or nodes that are specific to the current 
    robot configuration.
    - If you transition from fixed goals to following a dynamic human, you may need to adjust
    the `drl_vo_train.py` script and how it interacts with the `mix_cmd_vel` node or any other nodes
    responsible for interpreting dynamic goals.
 """


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch Arguments
    model_file = LaunchConfiguration('model_file') # Pretrained model file for DRL-VO training
    log_dir = LaunchConfiguration('log_dir') # Directory to save training logs and results

    # Path to other launch files
    velocity_smoother_launch_file = os.path.join(
        get_package_share_directory('turtlebot_teleop'),
        'launch', 'includes', 'velocity_smoother.launch.py'    # Velocity smoother for teleoperation
    )
    
    return LaunchDescription([
        # Declare Launch Arguments
        DeclareLaunchArgument('model_file', default_value=os.path.join(
            get_package_share_directory('drl_vo_nav'),
            'src', 'model', 'drl_pre_train.zip')), # Default pretrained model

        DeclareLaunchArgument('log_dir', default_value=os.path.join(
            get_package_share_directory('drl_vo_nav'),
            'src', 'runs')), # Default log directory

        # Include velocity smoother launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(velocity_smoother_launch_file)
        ),

        # DRL-VO Training Node
        Node(
            package='drl_vo_nav',
            executable='drl_vo_train.py',
            name='drl_vo_cmd',
            output='screen',
            parameters=[
                {'model_file': model_file},
                {'log_dir': log_dir}
            ],
        ),

        # Mixed Command Velocity Node
        Node(
            package='drl_vo_nav',
            executable='cmd_vel_pub.py',
            name='mix_cmd_vel',
            output='screen',
            remappings=[
                ('cmd_vel', 'teleop_velocity_smoother/raw_cmd_vel')
            ]
        ),
    ])
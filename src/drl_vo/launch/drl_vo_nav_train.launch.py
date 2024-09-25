"""
(Point to drl_vo_train.launch.py)

This file will define several arguments that control various aspects of the simulation and training
environment, such as:
- The scene file
- world name
- GUI options
- initial robot pose
- map file
- model file
- log directory
- whether vertain features (like OpenCV or RViz) are enabled.

The launch file will also define the nodes that are launched, such as:
- The Pedsim Gazebo
    + This section includes a launch file from the `pedsim_gazebo` package that will 
    start the simulation of pedestrians and the robot in the specified environment.

- The AMCL (Adaptive Monte Carlo Localization):
    + Includes the AMCL node launch file, which handles localization based on the provided map and initial robot pose.

- CNN data:
    + Includes a launch file responsible for setting up the CNN data processing pipeline.

- The DRL VO Control Policy:
    + Includes the training-specific launch file, where the DRL-VO model is trained using the
    provided model file and logs the results.

- RViz visualization:
    + Optionally includes RViz for visualizaing the navigation and training process if enabled.
"""


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory  # Import here
import os

def generate_launch_description():
    # Arguments
    scene_file = LaunchConfiguration('scene_file') # Scene file
    world_name = LaunchConfiguration('world_name') # World name
    gui = LaunchConfiguration('gui') # GUI options
    pose_initial_x = LaunchConfiguration('pose_initial_x') # Initial robot pose
    pose_initial_y = LaunchConfiguration('pose_initial_y') # Initial robot pose

    # map_file = LaunchConfiguration('map_file') # Map file
    
    # initial_pose_x = LaunchConfiguration('initial_pose_x') # Initial robot pose (AMCL)
    # initial_pose_y = LaunchConfiguration('initial_pose_y') # Initial robot pose (AMCL)
    # initial_pose_a = LaunchConfiguration('initial_pose_a') # Initial robot pose (AMCL)
    
    model_file = LaunchConfiguration('model_file')
    log_dir = LaunchConfiguration('log_dir')
    enable_opencv = LaunchConfiguration('enable_opencv')
    enable_console_output = LaunchConfiguration('enable_console_output')
    rviz = LaunchConfiguration('rviz')

    # Paths to other launch files
    pedsim_simulator_launch_file = os.path.join(
        get_package_share_directory('pedsim_simulator'), 
        'launch', 'robot.launch.py' # Launch file for the Pedsim Gazebo
    )
    
    # robot_gazebo_amcl_launch_file = os.path.join(
    #     get_package_share_directory('robot_gazebo'),
    #     'launch', 'amcl_demo_drl.launch.py'
    # )

    nav_cnn_data_launch_file = os.path.join(
        get_package_share_directory('drl_vo_nav'),
        'launch', 'nav_cnn_data.launch.py' # Launch file for the CNN data processing pipeline
    )

    drl_vo_train_launch_file = os.path.join(
        get_package_share_directory('drl_vo_nav'),
        'launch', 'drl_vo_train.launch.py' # Launch file for training the DRL-VO model
    )

    rviz_launch_file = os.path.join(
        get_package_share_directory('robot_gazebo'),
        'launch', 'view_navigation.launch.py' # Launch file for RViz visualization
    )

    return LaunchDescription([

        # Declare Launch Arguments
        DeclareLaunchArgument('scene_file', default_value=os.path.join(
            get_package_share_directory('pedsim_simulator'),
            'scenarios', 'eng_hall.xml')), # Default scene

        DeclareLaunchArgument('world_name', default_value=os.path.join(
            get_package_share_directory('pedsim_gazebo_plugin'),
            'worlds', 'eng_hall.world')), # Default world

        DeclareLaunchArgument('gui', default_value='true'), # Default GUI

        DeclareLaunchArgument('pose_initial_x', default_value='1.0'),
        DeclareLaunchArgument('pose_initial_y', default_value='1.0'),

        
        # DeclareLaunchArgument('map_file', default_value=os.path.join(
        #     get_package_share_directory('robot_gazebo'),
        #     'maps', 'gazebo_eng_lobby', 'gazebo_eng_lobby.yaml')), # Default map

        # DeclareLaunchArgument('initial_pose_x', default_value='1.0'),
        # DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        # DeclareLaunchArgument('initial_pose_a', default_value='0.13'),
        

        DeclareLaunchArgument('model_file', default_value=os.path.join(
            get_package_share_directory('drl_vo_nav'),
            'src', 'model', 'drl_pre_train.zip')), # Default model file

        DeclareLaunchArgument('log_dir', default_value=os.path.join(
            get_package_share_directory('drl_vo_nav'),
            'src', 'runs')), # Default log directory

        DeclareLaunchArgument('output', default_value='log'), # Default output
        DeclareLaunchArgument('enable_opencv', default_value='true'), # Default OpenCV
        DeclareLaunchArgument('enable_console_output', default_value='true'), # Default console output
        DeclareLaunchArgument('rviz', default_value='true'), # Default RViz

        # Include Launch Descriptions
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pedsim_simulator_launch_file),
            launch_arguments={
                'scene_file': scene_file,
                'world_name': world_name,
                'gui': gui,
                'pose_initial_x': pose_initial_x,
                'pose_initial_y': pose_initial_y
            }.items(), # Pass the arguments to the Pedsim Gazebo launch file
        ),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(robot_gazebo_amcl_launch_file),
        #     launch_arguments={
        #         'map_file': map_file,
        #         'initial_pose_x': initial_pose_x,
        #         'initial_pose_y': initial_pose_y,
        #         'initial_pose_a': initial_pose_a
        #     }.items(),
        # ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_cnn_data_launch_file), # Include the CNN data processing pipeline
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(drl_vo_train_launch_file),
            launch_arguments={
                'model_file': model_file,
                'log_dir': log_dir
            }.items(), # Pass the model file and log directory to the training launch file
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rviz_launch_file),
            condition=IfCondition(rviz), # Include RViz if enabled
        ),
    ])
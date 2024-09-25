from setuptools import setup

setup(
    name='turtlebot_gym',
    version='0.1',
    install_requires=[
        'gym',
        'rclpy',          # ROS 2 Python client library
        'numpy',          # Numerical computations
        'scipy',          # For optimization and scientific computing
    ],
    packages=['turtlebot_gym', 'turtlebot_gym.envs'],
    include_package_data=True,
    zip_safe=False
)
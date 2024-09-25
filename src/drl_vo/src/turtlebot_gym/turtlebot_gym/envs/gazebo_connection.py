#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64

class GazeboConnection(Node):

    def __init__(self, start_init_physics_parameters, reset_world_or_sim):
        super().__init__('gazebo_connection')

        # Create clients for Gazebo services
        self.unpause_client = self.create_client(Empty, '/gazebo/unpause_physics')
        self.pause_client = self.create_client(Empty, '/gazebo/pause_physics')
        self.reset_sim_client = self.create_client(Empty, '/gazebo/reset_simulation')
        self.reset_world_client = self.create_client(Empty, '/gazebo/reset_world')
        self.set_physics_client = self.create_client(SetPhysicsProperties, '/gazebo/set_physics_properties')

        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim

        self.init_values()
        self.pauseSim()

    def pauseSim(self):
        self.get_logger().info('Pausing Gazebo simulation...')
        self.call_service(self.pause_client)

    def unpauseSim(self):
        self.get_logger().info('Unpausing Gazebo simulation...')
        self.call_service(self.unpause_client)

    def resetSim(self):
        """
        Reset the simulation or the world based on the reset type.
        """
        if self.reset_world_or_sim == "SIMULATION":
            self.get_logger().info("Resetting SIMULATION...")
            self.call_service(self.reset_sim_client)
        elif self.reset_world_or_sim == "WORLD":
            self.get_logger().info("Resetting WORLD...")
            self.call_service(self.reset_world_client)
        else:
            self.get_logger().info("NO RESET SELECTED")

    def call_service(self, client):
        """
        Helper function to call a service client and wait for response.
        """
        client.wait_for_service(timeout_sec=5.0)
        request = Empty.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'Service call success: {client.srv_type}')
        else:
            self.get_logger().error(f'Service call failed: {client.srv_type}')

    def init_values(self):
        """
        Initialize the simulation physics parameters and reset the simulation.
        """
        self.resetSim()
        if self.start_init_physics_parameters:
            self.get_logger().info("Initializing Simulation Physics Parameters")
            self.init_physics_parameters()
        else:
            self.get_logger().info("Skipping Physics Parameters Initialization")

    def init_physics_parameters(self):
        """
        Initialize the physics parameters, like gravity, time step, and max update rate.
        """
        self._time_step = Float64()
        self._time_step.data = 0.001
        self._max_update_rate = Float64()
        self._max_update_rate.data = 1000.0

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = SetPhysicsPropertiesRequest().ode_config
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()

    def update_gravity_call(self):
        """
        Update the gravity and physics properties in the Gazebo world.
        """
        self.pauseSim()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        self.get_logger().info(f"Setting gravity to: {set_physics_request.gravity}")
        future = self.set_physics_client.call_async(set_physics_request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Gravity update result: {future.result().success}")
        else:
            self.get_logger().error("Failed to update gravity")

        self.unpauseSim()

    def change_gravity(self, x, y, z):
        """
        Change gravity dynamically.
        """
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z
        self.update_gravity_call()
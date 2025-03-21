import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandLong
from px4_msgs.msg import VehicleLandDetected
import numpy as np
from enum import Enum
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time

class State(Enum):
    IDLE = 0
    SEARCH = 1
    APPROACH = 2
    DESCEND_STEPWISE = 3
    LAND = 4
    FINISHED = 5

class ArucoTag:
    def __init__(self):
        self.position = np.full(3, np.nan)
        self.orientation = None
        self.timestamp = None

    def valid(self):
        return self.timestamp is not None

class PrecisionLand(Node):
    def __init__(self):
        super().__init__('precision_land')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self._target_pose_sub = self.create_subscription(
            PoseStamped, '/target_pose', self.target_pose_callback, qos_profile)
        self._vehicle_land_detected_sub = self.create_subscription(
            VehicleLandDetected, '/vehicle_land_detected', self.vehicle_land_detected_callback, 10)

        self._command_client = self.create_client(CommandLong, '/mavros/cmd/command')

        self._state = State.SEARCH
        self._tag = ArucoTag()
        self._land_detected = False
        self._precision_land_requested = False

        self._descent_altitudes = [8.0, 6.0, 4.0]  # Stepwise descent altitudes
        self._current_descent_index = 0
        self._last_descent_time = None

    def target_pose_callback(self, msg):
        self._tag.position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self._tag.orientation = msg.pose.orientation
        self._tag.timestamp = self.get_clock().now()

        self.get_logger().info(f"[TARGET POSE] Received: {self._tag.position}, Time: {self._tag.timestamp.nanoseconds}")

    def vehicle_land_detected_callback(self, msg):
        self._land_detected = msg.landed
        self.get_logger().info(f"[LAND DETECTED] Status: {self._land_detected}")

    def update_setpoint(self):
        if self._state == State.SEARCH:
            self.search_behavior()
        elif self._state == State.APPROACH:
            self.approach_behavior()
        elif self._state == State.DESCEND_STEPWISE:
            self.stepwise_descent_behavior()
        elif self._state == State.LAND:
            self.land_behavior()

    def search_behavior(self):
        if self._tag.valid():
            self.switch_to_state(State.APPROACH)

    def approach_behavior(self):
        if not self._tag.valid():
            self.switch_to_state(State.SEARCH)
            return

        if self._tag.position[2] <= 9.0:  # If detected within 10m
            self.switch_to_state(State.DESCEND_STEPWISE)
            self._last_descent_time = time.time()  # Start descent timer

    def stepwise_descent_behavior(self):
        current_time = time.time()

        if self._current_descent_index < len(self._descent_altitudes):
            target_altitude = self._descent_altitudes[self._current_descent_index]

            if (current_time - self._last_descent_time) >= 2.0:
                self.send_set_altitude(target_altitude)
                self._current_descent_index += 1
                self._last_descent_time = current_time  # Update timestamp

        else:
            self.switch_to_state(State.LAND)

    def land_behavior(self):
        if not self._precision_land_requested:
            self.send_mission_precision_land()
            self._precision_land_requested = True

        if self._land_detected:
            self.switch_to_state(State.FINISHED)

    def switch_to_state(self, state):
        self._state = state
        self.get_logger().info(f"[STATE CHANGE] Switched to: {state.name}")

    def send_set_altitude(self, target_altitude):
        if not self._command_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("[PX4] MAVROS command service unavailable")
            return

        req = CommandLong.Request()
        req.command = 84  # MAV_CMD_DO_CHANGE_ALTITUDE
        req.param1 = 0.0  # Reserved
        req.param2 = 0.0  # Reserved
        req.param3 = target_altitude  # New altitude
        req.param4 = 0.0  # Reserved
        req.param5 = 0.0  # Reserved
        req.param6 = 0.0  # Reserved
        req.param7 = 0.0  # Reserved

        self.get_logger().info(f"[PX4] Changing altitude to {target_altitude}m")

        future = self._command_client.call_async(req)
        future.add_done_callback(self.command_response_callback)

    def send_mission_precision_land(self):
        if not self._command_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("[PX4] MAVROS command service unavailable")
            return

        req = CommandLong.Request()
        req.command = 21  # MAV_CMD_NAV_LAND
        req.param1 = 0.0
        req.param2 = 2.0  # Precision Landing Type (2 = Required)
        req.param3 = 0.0
        req.param4 = 0.0
        req.param5 = float(self._tag.position[0])  # X Position
        req.param6 = float(self._tag.position[1])  # Y Position
        req.param7 = 0.0  # Target altitude (ignored for landing)

        self.get_logger().info("[PX4] Sending MAV_CMD_NAV_LAND for final descent.")

        future = self._command_client.call_async(req)
        future.add_done_callback(self.command_response_callback)

    def command_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("[PX4] Command executed successfully!")
            else:
                self.get_logger().error("[PX4] Command execution failed.")
        except Exception as e:
            self.get_logger().error(f"[PX4] Command service call failed: {str(e)}")

    def run(self):
        rate = self.create_rate(10)
        while rclpy.ok():
            self.update_setpoint()
            rclpy.spin_once(self, timeout_sec=0.1)

def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLand()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

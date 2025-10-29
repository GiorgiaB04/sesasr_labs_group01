import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf_transformations
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data

class BumpAndGo(Node):
    def __init__(self):
        super().__init__('bump_go_mod')

        # Declare parameters
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 1.5)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('min_distance_threshold', 0.6)

        # Read parameters
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.min_distance_threshold = self.get_parameter('min_distance_threshold').value

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.create_subscription(LaserScan, "scan", self.laser_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Timer
        self.timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)

        # Internal states
        self.scan_data = None
        self.current_yaw = 0.0
        self.state = 'GO_FORWARD'
        self.turn_direction = 1  # 1 = left, -1 = right
        self.position = Point()
        
        

    def odom_callback(self, msg):
        quat = msg.pose.pose.orientation
        quaternion = [quat.x, quat.y, quat.z, quat.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quaternion)
        self.current_yaw = yaw
        self.position = msg.pose.pose.position

    def laser_callback(self, msg):
        self.scan_data = msg

    def normalize_angle(self, angle):
        """ Normalize angle to be within [-pi, pi] """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def control_loop(self):
        if self.scan_data is None:
            return

        # Read front laser ranges (±10° for narrower detection)
        ### FIX: narrower detection
        

        # Example parameters (in your real code, these come from self.scan_data)
        # angle_min = self.scan_data.angle_min
        # angle_max = self.scan_data.angle_max
        # angle_increment = self.scan_data.angle_increment
        # ranges = self.scan_data.ranges

        angles = np.arange(self.scan_data.angle_min,
                        self.scan_data.angle_max,
                        self.scan_data.angle_increment)

        # If the number of angles doesn't exactly match ranges due to floating point issues:
        angles = angles[:len(self.scan_data.ranges)]

        # Define "front" sector in radians (e.g., ±15°)
        front_angle_width = np.deg2rad(15)

        # Select indices where the angle is near zero (front)
        front_indices = np.where(np.abs(angles) < front_angle_width)[0]

        # Get valid (non-infinite) readings
        valid_front = [(i, self.scan_data.ranges[i])
                    for i in front_indices
                    if not np.isinf(self.scan_data.ranges[i])]

        # Compute minimum distance in front
        min_front = min([dist for _, dist in valid_front], default=float('inf'))


      
        

        twist = Twist()

        # Dynamic obstacle threshold (smaller near goal)
        ### FIX
       
        dynamic_threshold = self.min_distance_threshold

        # --- STATE MACHINE ---
        if self.state == 'GO_FORWARD':
            if min_front < dynamic_threshold:
                self.get_logger().info('Obstacle detected! Switching to ROTATE')
                self.state = 'ROTATE'
                self.turn_direction = self.choose_turn_direction()
            else:
                twist.linear.x = self.linear_speed

        elif self.state == 'ROTATE':
            # Check if front is clear
            if min_front > dynamic_threshold + 0.1:
                self.get_logger().info('Path clear. Switching to GO_FORWARD')
                self.state = 'GO_FORWARD'
            else:
                twist.angular.z = self.turn_direction * self.angular_speed


        

        self.cmd_vel_pub.publish(twist)

    def choose_turn_direction(self):
        # Simple method: compare left and right average distances
        ranges = np.array(self.scan_data.ranges)
        left = ranges[60:100]
        right = ranges[260:300]
        left_avg = np.mean([r for r in left if not np.isinf(r)] or [float('inf')])
        right_avg = np.mean([r for r in right if not np.isinf(r)] or [float('inf')])
        return 1 if left_avg > right_avg else -1


def main(args=None):
    rclpy.init(args=args)
    node = BumpAndGo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

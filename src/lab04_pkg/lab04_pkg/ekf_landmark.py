import rclpy 
from rclpy.node import Node 
from nav_msgs.msg import Odometry 
from landmark_msgs.msg import LandmarkArray
import numpy as np 
import math 
import yaml 
from .EKF import RobotEKF 
from .velocity4ekf import velocity_motion_model_wrapper, jacobian_Gt, jacobian_Vt
from .landmark_model import landmark_model_jacobian



def load_landmarks(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    ids = data['landmarks']['id']
    xs = data['landmarks']['x']
    ys = data['landmarks']['y']

    positions = [[x, y] for x, y in zip(xs, ys)]

    return ids, positions


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        # ---- Load landmarks from YAML ----
        # NOTE: Verify this path is correct for your environment
        yaml_file = '/home/giorgia/group_101/src/lab04_pkg/lab04_pkg/landmarks_real.yaml'  
        self.landmark_ids, self.landmarks = load_landmarks(yaml_file)
        self.get_logger().info(f"Loaded {len(self.landmarks)} landmarks from YAML.")

        # ---- EKF setup ----
        self.ekf = RobotEKF()
        self.dim_x=3,
        self.dim_u=2,
        self.eval_gux=velocity_motion_model_wrapper,
        self.eval_Gx=jacobian_Gt,
        self.eval_Vu=jacobian_Vt, 

        # Tuning Parameters
        self.alpha = np.array([0.001, 0.01, 0.1, 0.2, 0.05, 0.05])
        self.ekf.Mt = np.eye(2)
        self.sigma_u = np.array([0.1, 0.1])
        self.sigma_z = np.array([0.3, math.pi / 24])
        self.Qt = np.diag(self.sigma_z**2)

        # Initial EKF uncertainty (will be overwritten upon initialization)
        self.ekf.Sigma = np.diag([0.5**2, 0.5**2, math.radians(10)**2])


        self.last_cmd = np.array([0.0, 0.0])
        self.last_time = self.get_clock().now()

        # Flags to initialize pose from first measurement
        self.initialized = False
        self.first_odom = None
        self.first_landmark = None

        # ---- ROS interfaces ----
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LandmarkArray, '/landmarks', self.landmarks_callback, 10)
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_timer(0.05, self.timer_callback)  # 20 Hz prediction

        self.get_logger().info('EKF node started with prediction + update.')
        if not self.initialized:
            self.first_odom = np.array([0.0, 0.77, 0.0])
            self.initialized=True

    # ----------------------------
    # Prediction callbacks
    # ----------------------------
    def odom_callback(self, msg):
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        self.last_cmd = np.array([v, omega])
            
    def timer_callback(self):
        now = self.get_clock().now() 
        dt = (now - self.last_time).nanoseconds / 1e9 
        self.last_time = now 
        
        if self.initialized:
            
            v = self.last_cmd[0]
            w = self.last_cmd[1]
            alpha = self.alpha
            
            # --- REQUIRED: Dynamic calculation of Mt ---
            v_var = alpha[0]*v**2 + alpha[1]*w**2
            w_var = alpha[2]*v**2 + alpha[3]*w**2
            
            self.ekf.Mt = np.diag([v_var, w_var])
            # ------------------------------------------

            # Predict new state and covariance
            self.ekf.predict(
                u=self.last_cmd,
                # The sigma_u argument is redundant if alpha is used in Mt, 
                # but if your EKF requires it, pass a placeholder like self.sigma_z
                sigma_u=self.sigma_u, 
                g_extra_args=(dt,),
            )
            
            self.get_logger().debug(f"EKF Prediction: dt={dt:.3f}s, u={self.last_cmd}, Mt_diag={self.ekf.Mt.diagonal()}")
       
        self.publish_estimated_state()

    # ----------------------------
    # Update callbacks
    # ----------------------------
    def landmarks_callback(self, msg: LandmarkArray):
        if not self.initialized:
            
            if self.first_odom is None:
                self.get_logger().warn("Skipping landmark: waiting for first odometry message.")
                return

            # Ensure we have a landmark measurement to initialize with
            if not msg.landmarks:
                 return

            lm = msg.landmarks[0]
            lm_id = lm.id

            # Lookup global (x,y) of this landmark from YAML
            if lm_id not in self.landmark_ids:
                self.get_logger().warn(f"Unknown landmark ID {lm_id}")
                return

            idx = self.landmark_ids.index(lm_id)
            landmark_pos = np.array(self.landmarks[idx])   # [x, y]

            # Odometry orientation (used as initial heading)
            q = self.first_odom.orientation
            # atan2(y, x) for quaternion (0, 0, z, w) is 2 * atan2(z, w)
            odom_theta = 2 * math.atan2(q.z, q.w)

            # Measurement
            r = lm.range
            b = lm.bearing

            # Robot global pose = landmark_global - measurement_in_global_frame
            robot_x = landmark_pos[0] - r * math.cos(odom_theta + b)
            robot_y = landmark_pos[1] - r * math.sin(odom_theta + b)

            self.ekf.mu = np.array([robot_x, robot_y, odom_theta])
            
            # --- MODIFICATION: Set larger initial covariance for robustness ---
            # Initial position error (e.g., 0.5m std dev)
            # Initial orientation error (e.g., 10 degrees std dev)
            self.ekf.Sigma = np.diag([0.5**2, 0.5**2, math.radians(10)**2]) 
            self.initialized = True
            # -----------------------------------------------------------------

            self.get_logger().info(f"EKF initialized in map frame: {self.ekf.mu}")
            return
        
        # Process EKF updates
        for lm in msg.landmarks:

            lm_id = lm.id
            if lm_id not in self.landmark_ids:
                self.get_logger().warn(f"Unknown landmark ID {lm_id}")
                continue
            
            idx = self.landmark_ids.index(lm_id)
            landmark_pos = np.array(self.landmarks[idx])

            z = np.array([lm.range, lm.bearing])

            self.ekf.update(
                z=z,
                eval_hx=self.eval_hx,
                eval_Ht=landmark_model_jacobian,
                Qt=self.Qt,
                Ht_args=(self.ekf.mu, landmark_pos),
                hx_args=(self.ekf.mu, landmark_pos),
            )
            

        
# ----------------------------
# Measurement model
# ----------------------------
    def eval_hx(self, mu, landmark):
        """Expected measurement ẑ = [range, bearing] from current state."""
        x, y, theta = mu
        m_x, m_y = landmark
        r_hat = math.sqrt((m_x - x) ** 2 + (m_y - y) ** 2)
        phi_hat = math.atan2(m_y - y, m_x - x) - theta
        # Normalize angle
        phi_hat = math.atan2(math.sin(phi_hat), math.cos(phi_hat))
        return np.array([r_hat, phi_hat])


    

    # ----------------------------
    # Publishing
    # ----------------------------
    def publish_estimated_state(self):
        # Only publish if initialized
        if not self.initialized:
            return
            
        mu = self.ekf.mu
        cov = self.ekf.Sigma

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = float(mu[0])
        msg.pose.pose.position.y = float(mu[1])
        msg.pose.pose.position.z = 0.0

        qz = math.sin(mu[2] / 2.0)
        qw = math.cos(mu[2] / 2.0)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Populate covariance matrix (only diagonal terms for simplicity)
        # 0: x-x, 7: y-y, 35: theta-theta
        msg.pose.covariance[0] = cov[0, 0]
        msg.pose.covariance[7] = cov[1, 1]
        msg.pose.covariance[35] = cov[2, 2]

        self.ekf_pub.publish(msg)
        self.get_logger().info(
            f"EKF Update → x={mu[0]:.2f}, y={mu[1]:.2f}, θ={mu[2]:.2f}"
        )

# ----------------------------
# Main
# ----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()




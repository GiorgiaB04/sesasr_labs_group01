import numpy as np
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from landmark_msgs.msg import LandmarkArray

from .ekf_landmark import EKFNode
from .EKF import RobotEKF
from .velocity4task2 import motion_model_wrapper
import yaml

def load_landmarks(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    ids = data['landmarks']['id']
    xs = data['landmarks']['x']
    ys = data['landmarks']['y']

    positions = [[x, y] for x, y in zip(xs, ys)]

    return ids, positions

class ExtendedEKFNode(EKFNode):
    """EKF with extended state: [x, y, theta, v, omega]."""
    
    def __init__(self):
        
        super().__init__()
        self.get_logger().info("Extended EKF: state = [x, y, θ, v, ω]")
        yaml_file = '/home/giorgia/ros2_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml'  
        self.landmark_ids, self.landmarks = load_landmarks(yaml_file)
        self.get_logger().info(f"Loaded {len(self.landmarks)} landmarks from YAML.")

        # --- CRITICAL FIX: RE-INITIALIZE EKF OBJECT WITH 5-STATE METHODS ---
        self.ekf = RobotEKF()
        self.ekf.dim_x = 5
        self.ekf.dim_u = 2  
        # ------------------------------------------------------------------
        self.ekf.eval_gux = motion_model_wrapper
        self.ekf.eval_Gt  = self.jacobian_Gt_extended
        self.ekf.eval_Vt  = self.jacobian_Vt_extended
    # 5-state Vt
        # Now set the 5-state initial conditions
        self.ekf.mu = np.zeros(5)     # x,y,theta,v,omega
        self.ekf.Sigma = np.diag([0.05, 0.05, 0.05, 0.2, 0.2])
        self.ekf._I = np.eye(5)

         
        self.alpha = np.array([0.05, 0.05])   # or your tuned values
        self.sigma_u = np.array([0.1, 0.1])   # noise on v and w ONLY


        # Measurement noise remains the same (just renamed for clarity)
        self.Qt_landmark = np.diag([0.3**2, (math.pi/24)**2])
        self.Qt_odom = np.diag([0.05**2, 0.05**2])
        self.Qt_imu  = np.diag([0.03**2])
      

       
       

        # Subscribe to IMU as well
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)

        self.get_logger().info("ExtendedEKFNode initialized.")
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

    
   



    def jacobian_Gt_extended(self, mu, u_unused, dt):

        x, y, th, v, w = mu

        return np.array([

        [1, 0, -v*math.sin(th)*dt, math.cos(th)*dt, 0],

        [0, 1, v*math.cos(th)*dt, math.sin(th)*dt, 0],

        [0, 0, 1, 0, dt],

        [0, 0, 0, 1, 0],

        [0, 0, 0, 0, 1]

        ])

    def jacobian_Vt_extended(self, mu, u_unused, dt):
        x, y, th, v, w = mu
        return np.array([
            [ math.cos(th)*dt,   0],
            [ math.sin(th)*dt,   0],
            [ 0,                 dt],
            [ 1,                 0],
            [ 0,                 1]
        ])
    def odom_callback(self, msg):
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        self.last_cmd = np.array([v, omega])

        if not self.initialized:
            self.first_odom = np.array([0.0, 0.77, 0.0])
    
    def timer_callback(self):
        now = self.get_clock().now() 
        dt = (now - self.last_time).nanoseconds / 1e9 
        self.last_time = now 
        
        # --- MODIFICATION: EKF PREDICTION STEP ADDED ---
        if self.initialized:
            
            # Predict new state and covariance
            self.ekf.predict(
                u=self.last_cmd,
                # Note: Assuming your RobotEKF uses these for prediction:
                sigma_u=self.sigma_u,
                # Pass necessary arguments for the motion model and Jacobians
                g_extra_args=(dt,),
                

                
            )
            
            self.get_logger().debug(f"EKF Prediction: dt={dt:.3f}s, u={self.last_cmd}")
        # ---------------------------------------------
       
        self.publish_estimated_state()
    

    # ============================================================
    #  Measurement Models
    # ============================================================

    # ---------- Landmarks → update x,y,θ ----------
    def h_landmark(self, mu, landmark):
        x, y, th, v, w = mu
        mx, my = landmark
        dx = mx - x
        dy = my - y
        r = math.sqrt(dx*dx + dy*dy)
        phi = math.atan2(dy,dx) - th
        return np.array([r, math.atan2(math.sin(phi), math.cos(phi))])

    def Ht_landmark(self, mu, landmark):
        x, y, th, v, w = mu
        mx, my = landmark
        dx = mx - x
        dy = my - y
        q = dx*dx + dy*dy
        sq = math.sqrt(q)
        return np.array([
            [-dx/sq, -dy/sq,   0, 0, 0],
            [ dy/q, -dx/q,  -1, 0, 0]
        ])

    # ---------- Wheel Encoder → update v,ω ----------
    def h_odom(self, mu):
        return mu[3:5]

    def Ht_odom(self, mu):
        return np.array([
            [0,0,0,1,0],
            [0,0,0,0,1]
        ])

    # ---------- IMU → update ω ----------
    def h_imu(self, mu):
        return np.array([mu[4]])

    def Ht_imu(self, mu):
        return np.array([[0,0,0,0,1]])

    # ============================================================
    #  Update Callbacks
    # ============================================================

    

    def imu_callback(self, msg):
        """IMU angular velocity → updates omega only."""
        omega_meas = msg.angular_velocity.z
        z = np.array([omega_meas])

        self.ekf.update(
            z=z,
            eval_hx=self.h_imu,
            eval_Ht=self.Ht_imu,
            Qt=self.Qt_imu,
            Ht_args=(self.ekf.mu,),
            hx_args=(self.ekf.mu,)
        )
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
            v0 = self.last_cmd[0]
            w0 = self.last_cmd[1]

            self.ekf.mu = np.array([robot_x, robot_y, odom_theta, v0,w0])
            
            # --- MODIFICATION: Set larger initial covariance for robustness ---
            # Initial position error (e.g., 0.5m std dev)
            # Initial orientation error (e.g., 10 degrees std dev)
            self.ekf.Sigma = np.diag([0.5**2, 0.5**2, math.radians(10)**2,0.2**2,0.2**2]) 
            self.initialized = True
            # -----------------------------------------------------------------

            self.get_logger().info(f"EKF initialized in map frame: {self.ekf.mu}")
            return

def main(args=None):
    rclpy.init(args=args)
    node = ExtendedEKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


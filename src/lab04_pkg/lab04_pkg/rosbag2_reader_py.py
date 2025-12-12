import numpy as np
import math
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# --- METRICS FUNCTIONS ---

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean((actual - predicted)**2)

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def mae(error: np.ndarray):
    """ Mean Absolute Error """
    
    return np.mean(np.abs(error))



class Rosbag2Reader:
    """
    A class to easily iterate over the messages in a ROS 2 bag
    ... (rest of the class implementation) ...
    """

    def __init__(self, path, topics_filter=[], storage_id="sqlite3", serialization_format="cdr"):
        self.__path = path
        self.__set_rosbag_options(storage_id, serialization_format)
        self.__reader = rosbag2_py.SequentialReader()
        self.__reader.open(self.__storage_options, self.__converter_options)

        topic_types = self.__reader.get_all_topics_and_types()
        self.__type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        self.set_filter(topics_filter)

    def __iter__(self):
        self.__reset_bag_reader()
        return self

    def __next__(self):
        if self.__reader.has_next():
            (topic, data, t) = self.__reader.read_next()
            msg_type = get_message(self.__type_map[topic])
            msg = deserialize_message(data, msg_type)
            return (topic, msg, t)
        else:
            raise StopIteration

    # ... (other Rosbag2Reader methods: path, all_topics, selected_topics, set_filter, reset_filter, __set_rosbag_options, __reset_bag_reader) ...
    
    @property
    def path(self):
        return self.__path

    @property
    def all_topics(self):
        return self.__type_map

    @property
    def selected_topics(self):
        if self.__storage_filter is None:
            return self.all_topics
        else:
            return self.__selected_topics            

    def set_filter(self, topics):
        if topics:
            try:
                self.__selected_topics = {topic: self.__type_map[topic] for topic in topics}
            except KeyError as e:
                raise KeyError(f"Could not find topic {e} in the rosbag file")
            self.__storage_filter = rosbag2_py.StorageFilter(topics=topics)
        else:
            self.__storage_filter = None

        self.__reset_bag_reader()

    def reset_filter(self):
        self.__storage_filter = None
        self.__reader.reset_filter()
        self.__reset_bag_reader()

    def __set_rosbag_options(self, storage_id, serialization_format):
        self.__storage_options = rosbag2_py.StorageOptions(uri=self.__path, storage_id=storage_id)

        self.__converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format,
        )

    def __reset_bag_reader(self):
        self.__reader.open(self.__storage_options, self.__converter_options)
        if self.__storage_filter is not None:
            self.__reader.set_filter(self.__storage_filter)


# --- UTILITY FUNCTION ---
def quaternion_to_yaw(q):
    """Convert a geometry_msgs/Quaternion to yaw angle (radians)"""
    # Assuming q is a geometry_msgs/Quaternion object with x, y, z, w fields
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_ekf_metrics(bag_path, ekf_topic, truth_topic):
    """
    Computes RMSE and MAE between EKF output and ground truth.

    :param bag_path: Path to the ROS 2 bag folder.
    :param ekf_topic: Topic publishing the EKF's estimated pose (e.g., '/ekf/pose').
    :param truth_topic: Topic publishing the ground truth pose (e.g., '/ground_truth/pose').
    """
    try:
        # 1. Initialize Reader and Filter Topics
        reader = Rosbag2Reader(bag_path, topics_filter=[ekf_topic, truth_topic])
        print(f"Reading bag: {bag_path}")
        print(f"EKF Topic: {ekf_topic}, Truth Topic: {truth_topic}")

    except Exception as e:
        print(f"Error initializing Rosbag2Reader: {e}")
        return

    # Data structures to store time-synchronized poses
    ekf_data = {}    # {timestamp: [x, y, theta]}
    truth_data = {}  # {timestamp: [x, y, theta]}

    # 2. Extract Data
    # The EKF and truth topics are typically nav_msgs/Odometry or geometry_msgs/PoseStamped
    for topic, msg, t in reader:
        # Extract X, Y, and Yaw
        if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'): # Odometry or PoseStamped
            pose = msg.pose.pose
        elif hasattr(msg, 'pose'): # geometry_msgs/Pose
            pose = msg.pose
        else:
            continue # Skip non-pose messages

        x = pose.position.x
        y = pose.position.y
        theta = quaternion_to_yaw(pose.orientation)
        
        # Store data, keyed by timestamp
        if topic == ekf_topic:
            ekf_data[t] = np.array([x, y, theta])
        elif topic == truth_topic:
            truth_data[t] = np.array([x, y, theta])

    # 3. Time Synchronization and Error Calculation
    
    # Find common and near-common timestamps to ensure fair comparison
    # We use the timestamps from the ground truth as the reference timeline
    
    truth_timestamps = sorted(truth_data.keys())
    
    # Store synchronized data arrays
    synced_truth = []
    synced_ekf = []
    
    # Tolerance for time synchronization (e.g., 5 ms)
    time_tolerance = 5 * 10**6 

    for t_truth in truth_timestamps:
        # Find the closest EKF timestamp
        closest_t_ekf = min(ekf_data.keys(), key=lambda t_ekf: abs(t_ekf - t_truth))
        
        if abs(t_truth - closest_t_ekf) <= time_tolerance:
            synced_truth.append(truth_data[t_truth])
            synced_ekf.append(ekf_data[closest_t_ekf])

    if not synced_truth:
        print("Error: Could not synchronize any data points between EKF and Ground Truth topics.")
        return

    # Convert lists of [x, y, theta] arrays to single NumPy arrays
    synced_truth = np.array(synced_truth)
    synced_ekf = np.array(synced_ekf)
    
    # Calculate error matrix: Error = Actual - Predicted
    error_matrix = synced_truth - synced_ekf

    # Normalize angular error (theta) to the range (-pi, pi]
    error_matrix[:, 2] = np.arctan2(np.sin(error_matrix[:, 2]), np.cos(error_matrix[:, 2]))
    
    # 4. Compute and Print Metrics
    
    # Separate data for X, Y, and Theta
    truth_x, truth_y, truth_theta = synced_truth.T
    ekf_x, ekf_y, ekf_theta = synced_ekf.T
    error_x, error_y, error_theta = error_matrix.T
    
    # Combined Position Error (Euclidean Distance)
    pos_error = np.sqrt(error_x**2 + error_y**2)
    
    print("\n--- EKF Metric Results ---")
    print(f"Total synchronized samples: {len(synced_truth)}")
    print("--------------------------")
    
    print("POSITION METRICS (X and Y combined):")
    print(f"  RMSE (Position): {rmse(pos_error, np.zeros_like(pos_error)):.4f} m")
    print(f"  MAE (Position):  {mae(pos_error):.4f} m")
    
    print("\nINDIVIDUAL METRICS:")
    print(f"  RMSE (X):        {rmse(truth_x, ekf_x):.4f} m")
    print(f"  MAE (X):         {mae(error_x):.4f} m")
    
    print(f"  RMSE (Y):        {rmse(truth_y, ekf_y):.4f} m")
    print(f"  MAE (Y):         {mae(error_y):.4f} m")

    print(f"  RMSE (Theta):    {rmse(truth_theta, ekf_theta):.4f} rad")
    print(f"  MAE (Theta):     {mae(error_theta):.4f} rad")
    print("--------------------------")


def main():
   
    BAG_PATH = '/home/giorgia/group_101/src/rosbags/task2_sim/Qt=0.08^2' 
    EKF_TOPIC = '/ekf'           
    TRUTH_TOPIC = '/ground_truth'     

    compute_ekf_metrics(BAG_PATH, EKF_TOPIC, TRUTH_TOPIC)

if __name__ == '__main__':
    main()

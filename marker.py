import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandTOL

class ArucoLanding(Node):
    def __init__(self):
        super().__init__('aruco_landing')
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_id = 0  # Target ArUco marker ID
        self.marker_size = 100  # mm
        self.cap = cv2.VideoCapture(0)
        self.landing_client = self.create_client(CommandTOL, '/mavros/cmd/land')
    
    def detect_aruco_markers(self):
        while rclpy.ok() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Failed to grab frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                for i, detected_id in enumerate(ids):
                    if detected_id[0] == self.marker_id:
                        self.get_logger().info(f"Aruco Marker {self.marker_id} detected. Initiating landing...")
                        self.initiate_landing()
                        break
            
            cv2.imshow('ArUco Marker Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def initiate_landing(self):
        if not self.landing_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Landing service unavailable")
            return
        
        request = CommandTOL.Request()
        request.altitude = 0.0
        request.latitude = 0.0
        request.longitude = 0.0
        request.min_pitch = 0.0
        request.yaw = 0.0
        
        future = self.landing_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info("Landing command sent successfully")
        else:
            self.get_logger().error("Landing command failed")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoLanding()
    node.detect_aruco_markers()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

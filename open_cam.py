import cv2
import cv2.aruco as aruco

def detect_aruco_markers():
    # Load the ArUco dictionary with 7x7 markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Define marker ID and size
    marker_id = 0
    marker_size = 100  # mm
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Draw markers and display IDs
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    c = corners[i][0]
                    center_x = int((c[0][0] + c[2][0]) / 2)
                    center_y = int((c[0][1] + c[2][1]) / 2)
                    cv2.putText(frame, f'ID: {detected_id[0]} Size: {marker_size}mm', 
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('ArUco Marker Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    detect_aruco_markers()

if __name__ == "__main__":
    main()

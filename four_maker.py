import cv2
import cv2.aruco as aruco
import numpy as np

def detect_aruco_markers():
    # Load the ArUco dictionary with 7x7 markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Define marker IDs and size
    marker_ids = [0, 1, 2, 3]
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
        
        # Store marker centers
        marker_centers = []
        
        # Draw markers and display IDs
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, detected_id in enumerate(ids):
                if detected_id[0] in marker_ids:
                    c = corners[i][0]
                    center_x = int((c[0][0] + c[2][0]) / 2)
                    center_y = int((c[0][1] + c[2][1]) / 2)
                    marker_centers.append((center_x, center_y))
                    cv2.putText(frame, f'ID: {detected_id[0]}', (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate and draw square center if all markers are detected
        if len(marker_centers) == 4:
            avg_x = int(np.mean([p[0] for p in marker_centers]))
            avg_y = int(np.mean([p[1] for p in marker_centers]))
            cv2.circle(frame, (avg_x, avg_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, 'Center', (avg_x + 10, avg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('ArUco Marker Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_markers()
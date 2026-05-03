import cv2
import numpy as np
import mediapipe as mp

mp_faces = mp.solutions.face_detection
cap = cv2.VideoCapture(0)

# --- Configuration ---
SMOOTH_WEIGHT = 0.2    # 0.1 for very slow/smooth, 0.5 for fast follow
MAX_LOST_FRAMES = 10   # How many frames to "remember" a face after it's gone
trackers = {}          # Dictionary to store {face_id: [curr_x, curr_y, curr_w, curr_h, lost_count]}

def apply_cctv_effect(frame):
    noise = np.random.randint(0, 12, frame.shape, dtype='uint8')
    frame = cv2.addWeighted(frame, 0.93, noise, 0.07, 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.4 
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[::3, :] = frame[::3, :] * 0.9
    return frame

with mp_faces.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        display_frame = apply_cctv_effect(frame.copy())
        ih, iw, _ = frame.shape
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        active_this_frame = set()

        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                tx, ty, tw, th = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                 int(bbox.width * iw), int(bbox.height * ih)
                
                # Simple ID based on index (works for basic multi-face)
                face_id = i 
                active_this_frame.add(face_id)

                if face_id not in trackers:
                    # Initialize tracker for new face
                    trackers[face_id] = [tx, ty, tw, th, 0]
                else:
                    # Update existing tracker with Smoothing (Lerp)
                    curr = trackers[face_id]
                    curr[0] += (tx - curr[0]) * SMOOTH_WEIGHT
                    curr[1] += (ty - curr[1]) * SMOOTH_WEIGHT
                    curr[2] += (tw - curr[2]) * SMOOTH_WEIGHT
                    curr[3] += (th - curr[3]) * SMOOTH_WEIGHT
                    curr[4] = 0 # Reset lost counter

        # Clean up or handle lost faces
        to_delete = []
        for fid, data in trackers.items():
            if fid not in active_this_frame:
                data[4] += 1 # Increment lost counter
                if data[4] > MAX_LOST_FRAMES:
                    to_delete.append(fid)
            
            # Only draw if the face isn't "too lost"
            if data[4] <= MAX_LOST_FRAMES:
                x, y, w, h = map(int, data[:4])
                
                # Draw main bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

                # Add zoom inset ONLY for the very first/prominent face detected
                if fid == 0:
                    pad = 60
                    y1, y2, x1, x2 = max(0, y-pad), min(ih, y+h+pad), max(0, x-pad), min(iw, x+w+pad)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        zoom = cv2.resize(crop, (250, 250))
                        zoom = cv2.detailEnhance(zoom, sigma_s=10, sigma_r=0.15)
                        ix, iy = iw - 280, 30
                        cv2.line(display_frame, (x + w, y), (ix, iy + 125), (255, 255, 255), 1)
                        display_frame[iy:iy+250, ix:ix+250] = zoom
                        cv2.rectangle(display_frame, (ix, iy), (ix+250, iy+250), (255, 255, 255), 2)

        for fid in to_delete:
            del trackers[fid]

        cv2.imshow('CivisCams', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
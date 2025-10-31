"""
Footfall Counter using Computer Vision
A system that detects, tracks, and counts people crossing a virtual line in a video.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import argparse
from pathlib import Path

class CentroidTracker:
    """
    Simple centroid-based tracker for people detection.
    Assigns unique IDs to objects and tracks them across frames.
    """
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # ID: centroid
        self.disappeared = {}  # ID: frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories = defaultdict(lambda: deque(maxlen=30))
        
    def register(self, centroid):
        """Register a new object with unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove object that has disappeared"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]
            
    def update(self, detections):
        """
        Update tracker with new detections
        Returns: dictionary of object_id: centroid
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(detections):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distances between existing and new centroids
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(obj_centroid - input_centroid)
            
            # Match objects to detections
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(input_centroids[col])
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            
            # Handle disappeared objects
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects


class FootfallCounter:
    """
    Main footfall counter class that handles detection, tracking, and counting.
    """
    def __init__(self, video_source, line_position=0.5, confidence_threshold=0.5):
        self.video_source = video_source
        self.line_position = line_position  # Relative position (0-1) for counting line
        self.confidence_threshold = confidence_threshold
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate counting line position
        self.line_y = int(self.height * line_position)
        
        # Initialize tracker
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=100)
        
        # Counting variables
        self.entry_count = 0
        self.exit_count = 0
        self.crossed = {}  # Track which objects crossed the line
        self.previous_positions = {}  # Track previous y-positions
        
        # Load YOLO model (using OpenCV DNN)
        self.net = self._load_yolo_model()
        self.output_layers = self._get_output_layers()
        
    def _load_yolo_model(self):
        """Load YOLOv4-tiny model using OpenCV DNN"""
        try:
           
            net = cv2.dnn.readNet("models/yoloy4-tiny.weights", "models/yolo4-tiny.cfg")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return net
        except:
            print("Warning: Could not load YOLO model. Using HOG detector instead.")
            return None
    
    def _get_output_layers(self):
        """Get YOLO output layer names"""
        if self.net is None:
            return None
        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect_people_yolo(self, frame):
        """Detect people using YOLO"""
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Class 0 is 'person' in COCO dataset
                if class_id == 0 and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(boxes[i])
        
        return detections
    
    def detect_people_hog(self, frame):
        """Fallback: Detect people using HOG descriptor"""
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        detections = []
        for (x, y, w, h) in rects:
            detections.append([x, y, w, h])
        
        return detections
    
    def count_crossing(self, object_id, centroid):
        """
        Determine if a person crossed the line and update counts.
        Logic: Track if person moves from above line to below (entry) or vice versa (exit)
        """
        cy = centroid[1]
        
        if object_id not in self.previous_positions:
            self.previous_positions[object_id] = cy
            return
        
        prev_y = self.previous_positions[object_id]
        
        # Check if crossed the line
        if object_id not in self.crossed:
            # Crossing downward (entry)
            if prev_y < self.line_y and cy >= self.line_y:
                self.entry_count += 1
                self.crossed[object_id] = 'entry'
            # Crossing upward (exit)
            elif prev_y > self.line_y and cy <= self.line_y:
                self.exit_count += 1
                self.crossed[object_id] = 'exit'
        
        self.previous_positions[object_id] = cy
    
    def draw_info(self, frame, objects):
        """Draw bounding boxes, line, and counts on frame"""
        # Draw counting line
        cv2.line(frame, (0, self.line_y), (self.width, self.line_y), (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw object centroids and IDs
        for object_id, centroid in objects.items():
            text = f"ID: {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, tuple(centroid), 4, (0, 255, 0), -1)
            
            # Draw trajectory
            if object_id in self.tracker.trajectories:
                points = list(self.tracker.trajectories[object_id])
                for i in range(1, len(points)):
                    cv2.line(frame, tuple(points[i-1]), tuple(points[i]), (0, 200, 0), 2)
        
        # Draw counts
        info_text = [
            f"Entry Count: {self.entry_count}",
            f"Exit Count: {self.exit_count}",
            f"Total: {self.entry_count + self.exit_count}",
            f"Current People: {len(objects)}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (5, y_offset + i * 30 - 25), 
                         (400, y_offset + i * 30 + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, output_path=None, display=True):
        """Main processing loop"""
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                    (self.width, self.height))
        
        frame_count = 0
        
        print("Processing video... Press 'q' to quit early.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect people
            if self.net is not None:
                detections = self.detect_people_yolo(frame)
            else:
                detections = self.detect_people_hog(frame)
            
            # Update tracker
            objects = self.tracker.update(detections)
            
            # Count crossings
            for object_id, centroid in objects.items():
                self.count_crossing(object_id, centroid)
            
            # Visualize
            frame = self.draw_info(frame, objects)
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Display
            if display:
                cv2.imshow("Footfall Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames | Entry: {self.entry_count} | Exit: {self.exit_count}")
        
        # Cleanup
        self.cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print final results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Entry Count: {self.entry_count}")
        print(f"Exit Count: {self.exit_count}")
        print(f"Total Crossings: {self.entry_count + self.exit_count}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Footfall Counter using Computer Vision')
    parser.add_argument('--video', type=str, required=True, help='Path to input video or camera index (0)')
    parser.add_argument('--output', type=str, default=None, help='Path to output video')
    parser.add_argument('--line', type=float, default=0.5, help='Line position (0-1, default: 0.5)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    
    args = parser.parse_args()
    
    # Handle camera input
    video_source = args.video
    if args.video.isdigit():
        video_source = int(args.video)
    
    # Create counter
    counter = FootfallCounter(
        video_source=video_source,
        line_position=args.line,
        confidence_threshold=args.confidence
    )
    
    # Process video
    counter.process_video(
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
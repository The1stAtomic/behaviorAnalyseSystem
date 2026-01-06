import time
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path

def load_config():
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.absolute()}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Config file is empty or invalid: {config_path.absolute()}")
        return config

#Stub components

def estimate_head_direction(student_crop):
  """
    Phase 3 stub.
    Later replaced with real head pose estimation.
  """
  return "unknown"

def iou(boxA, boxB):
    """
    Compute Intersection over Union between two boxes.
    Boxes: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0.0

    return interArea / (boxAArea + boxBArea - interArea)
  
#Main

def main():
  config = load_config()
  
  camera_index = config["camera"]["index"]
  frame_sample_rate = config["video"]["frame_sample_rate"]
  iou_threshold = config["fusion"]["iou_threshold"]
  
  model = YOLO(config["detection"]["model"])
  
  cap = cv2.VideoCapture(camera_index)
  if not cap.isOpened():
      print("Error: Could not open video.")
      return
    
  frame_count = 0
  start_time = time.time()
  
  print("Pipeline started")
  
  while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
      
    frame_count += 1
    if frame_count % frame_sample_rate != 0:
        continue
      
    timestamp = time.time() - start_time
    
    #Step 3-4: Detetction and Tracking
    results = model.track(
      frame,
      persist=config["tracking"]["persist"],
      classes=[config["detection"]["person_class_id"]],  # Person class
      verbose=False
    )
    
    if results[0].boxes.id is None:
        continue
      
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    
    #Step 5: Head Direction Estimation (Stub)
    students = []
    for box, track_ids in zip(boxes, track_ids):
      x1, y1, x2, y2 = map(int, box)
      student_crop = frame[y1:y2, x1:x2]
      
      head_dir = estimate_head_direction(student_crop)
      
      students.append({
        "track_id": track_ids,
        "bbox": box,
        "head_direction": head_dir
      })
      
    #Step 6: Phone detection(stubbed as empty)
    phone_boxes = []  # Placeholder for phone detection results
    
    #Step 7: Signal fusion
    for student in students:
      phone_overlap = False
      for phone_box in phone_boxes:
        if iou(student["bbox"], phone_box) > iou_threshold:
          phone_overlap = True
          break
      
      #Output (Phase 3 logging)
      print(
        f"[{timestamp:6.2f}s] "
        f"Student #{student['track_id']} | "
        f"head={student['head_direction']} | "
        f"phone={phone_overlap}"
      )
      
      #Debug draw
      x1, y1, x2, y2 = map(int, student["bbox"])
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(
        frame,
        f"ID {student['track_id']}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 0), 2
      )
      
    if config["debug"]["show_window"]:
      cv2.imshow("Debug View", frame)
      if cv2.waitKey(1) & 0xFF == config["debug"]["exit_key"]:
        break
      
  cap.release()
  cv2.destroyAllWindows()
  print("Pipeline ended")
  
if __name__ == "__main__":
  main()
import cv2
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from tkinter import Tk, filedialog
from PIL import Image
import numpy as np
import os

# Disable Tkinter root window
Tk().withdraw()

# Open file picker
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
if not video_path:
    print("No video selected. Exiting.")
    exit()

# Get the directory of the selected video file
video_dir = os.path.dirname(video_path)

# Define the path for the detection log file in the same directory
log_file_path = os.path.join(video_dir, "detection_log.txt")

# Load the OWL-ViT model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define your custom non-COCO object classes
prompts = ["a lightbulb", "a matchstick", "a monitor", "a lion", "a gaming console"]

# Open the video file
cap = cv2.VideoCapture(video_path)

# Open text file for real-time logging
with open(log_file_path, "a") as log_file:
    log_file.write("Detection Results Log:\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the input
        inputs = processor(text=prompts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the predictions
        target_size = torch.Tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_grounded_object_detection(outputs, target_sizes=target_size, threshold=0.1)[0]

        # Draw results on the frame and log them in real-time
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{prompts[label]}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Write detected objects to text file in real-time
            log_file.write(f"Detected: {prompts[label]} with confidence: {score:.2f}\n")

        # Show the frame
        cv2.imshow("Zero-Shot Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

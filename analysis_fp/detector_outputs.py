import pandas as pd
from ultralytics import YOLO
import os

def run_object_detection(image_files, model_path="yolo11n.pt", output_dir="output/detector_ops"):
    """
    Perform object detection on a list of images using YOLO and save results to CSV files.

    Parameters:
    - image_files (list): List of image file paths.
    - model_path (str): Path to the YOLO model file.
    - output_dir (str): Directory to save detection result CSV files.

    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each image and perform object detection
    for image_path in image_files:
        # Perform object detection on an image
        results = model(image_path)
        
        # Define confidence thresholds
        confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Prepare a list to hold results for the current image
        image_detections = []
        
        # Iterate over each confidence threshold
        for result in results:
            # Access detection boxes
            boxes = result.boxes
            
            # If there are no detections, continue to the next iteration
            if boxes is None:
                continue
            
            # Convert boxes to a DataFrame
            detections = boxes.xyxy.cpu().numpy()  # Get bounding box coordinates as a NumPy array
            confidences = boxes.conf.cpu().numpy()  # Get confidence scores as a NumPy array
            class_ids = boxes.cls.cpu().numpy()      # Get class IDs as a NumPy array

            # Create a DataFrame for current detections
            df = pd.DataFrame(detections, columns=['x1', 'y1', 'x2', 'y2'])
            df['confidence'] = confidences
            df['class_id'] = class_ids

            # Add the image path to the column
            df['image_path'] = len(df) * [image_path]

            # Add detections to the current image list
            image_detections.append(df)
        
        # Concatenate all detections into a single DataFrame if there are any detections
        if image_detections:
            final_results = pd.concat(image_detections)
            # Construct the output file path for the current image
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{image_name}.csv")
            # Save results to the CSV file in the output directory
            final_results.to_csv(output_path, index=False)
        else:
            print(f"No detections found for {image_path}.")

    print(f"Detection results saved to '{output_dir}' directory.")


image_files = ["cg.png"]
run_object_detection(image_files)

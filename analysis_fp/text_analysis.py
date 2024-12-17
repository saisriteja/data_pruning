import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

class ImageCaptioner:
    def __init__(self):
        # Initialize the BLIP processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    def crop_image(self, image, x1, y1, x2, y2, buffer=20):
        # Expand the bounding box with a margin or buffer
        x1 = max(0, x1 - buffer)
        y1 = max(0, y1 - buffer)
        x2 = min(image.width, x2 + buffer)
        y2 = min(image.height, y2 + buffer)
        return image.crop((x1, y1, x2, y2))

    def generate_caption(self, image_path, x1, y1, x2, y2):
        # Open the image
        image = Image.open(image_path)
        
        # Crop the image with a buffer
        cropped_image = self.crop_image(image, x1, y1, x2, y2)
        
        # Run the BLIP model on the cropped image
        inputs = self.processor(cropped_image, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption


def process_csv_files(csv_files, caption_output_dir="output/caption"):
    """
    Perform image caption analysis on a list of CSV files and save the results with captions.

    Parameters:
    - csv_files (list): List of CSV file paths.
    - caption_output_dir (str): Directory to save captioned CSV files.

    """
    # Ensure the caption output directory exists
    os.makedirs(caption_output_dir, exist_ok=True)

    # Create an instance of the ImageCaptioner class
    captioner = ImageCaptioner()

    for csv_file in csv_files:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Prepare data for the new CSV file
        new_data = []

        for index, row in df.iterrows():
            image_path = row['image_path']
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            
            # Generate caption for the cropped image with expanded bounding box
            caption = captioner.generate_caption(image_path, x1, y1, x2, y2)
            
            # Append data to the new CSV list
            new_data.append([x1, y1, x2, y2, row['confidence'], row['class_id'], 20, image_path, caption])

        # Create a new DataFrame for the new CSV file
        new_df = pd.DataFrame(new_data, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'threshold', 'image_path', 'caption'])

        # Save the new DataFrame to a CSV file in the output directory
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        caption_file_path = os.path.join(caption_output_dir, f"{image_name}.csv")
        new_df.to_csv(caption_file_path, index=False)

        print(f"New CSV file with captions saved as {caption_file_path}")


# Example usage:
csv_files = ['output/detector_ops/cg.csv']#, 'detector_ops/another_detection.csv']  # List of CSV files
process_csv_files(csv_files)

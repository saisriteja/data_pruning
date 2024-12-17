from cgitb import text
import os
import pandas as pd
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# Define the CLIP processor and model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the statements dictionary for context
statements = {
    "Viewpoint Variation": {
        "Perspective/Angle": ["The object is viewed from a frontal angle.", "The object is viewed from a side angle.", "The object is viewed from an oblique angle."],
        "Distance": ["The object is far from the camera.", "The object is close to the camera.", "The object is at medium distance."],
        "Orientation": ["The object is tilted.", "The object is rotated."],
        "Distance/Zoom Level": ["The object is zoomed in.", "The object is zoomed out.", "The object is at normal zoom level."]
    },
    "Occlusion": {
        "Partial Occlusion": ["Some part of the object is occluded by another object.", "Some part of the object is occluded by the background."],
        "Occluder Type": ["The occluder is an object.", "The occluder is a shadow."],
        "Occlusion Depth": ["The object is partially occluded.", "The object is fully occluded.", "The object is only background."]
    },
    "Illumination Conditions": {
        "Lighting Source": ["The primary lighting source is natural light.", "The primary lighting source is artificial light."],
        "Shadowing": ["There are shadows cast on the object or background."],
        "Lighting Intensity": ["The scene is well-lit.", "The scene is underexposed.", "The scene is overexposed."],
        "Color Temperature": ["The color temperature is warm.", "The color temperature is neutral.", "The color temperature is cool."]
    },
    "Cluttered or Textured Backgrounds": {
        "Background Clutter": ["The background is complex with multiple objects.", "The background is simple with few objects."],
        "Background Texture": ["The background is smooth.", "The background is rough.", "The background is patterned."],
        "Foreground-Background Relationship": ["The object is integrated with the background.", "The object is separate from the background.", "The object is detached from the background."]
    },
    "Deformation and Aspect Ratio Variation": {
        "Shape Deformation": ["The object is distorted or deformed due to perspective or angle."],
        "Aspect Ratio": ["The aspect ratio is compressed.", "The aspect ratio is stretched.", "The aspect ratio is normal."],
        "Size/Scale": ["The object is scaled down.", "The object is scaled up.", "The object is at normal size."]
    },
}




# import os
# import pandas as pd
# from PIL import Image

# # Directory to save cropped images
# output_dir = "output/cropped_images"
# os.makedirs(output_dir, exist_ok=True)

# # Read the CSV file
# csv_file = 'output/detector_ops/cg.csv'
# df = pd.read_csv(csv_file)


# # Loop through each bounding box in the CSV file
# for index, row in df.iterrows():
#     image_path = row['image_path']
#     x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
    
#     # Load the image
#     image = Image.open(image_path)
    
#     # Crop the image using bounding box coordinates
#     cropped_image = image.crop((x1, y1, x2, y2))



#     labels = dict()

#     # Print each list one by one
#     for category, variations in statements.items():
#         # print(f"Category: {category}")
#         for sub_category, questions in variations.items():
#             # print(f"  {sub_category}:")

#             inputs = processor(text=questions, images=cropped_image, return_tensors="pt", padding=True)
#             outputs = model(**inputs)
#             logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#             probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

#             # get the position maxiumn probability
#             max_prob = probs[0][0].item()

#             # get its position
#             max_prob_pos = probs[0].argmax().item()

#             labels[sub_category] = max_prob_pos

    
#     print(labels)
#     quit()



import os
import pandas as pd
from PIL import Image


from tqdm import tqdm

def process_csv_files(csv_file_list, output_dir, statements, processor, model):
    """
    Processes a list of CSV files to crop images and generate labels.
    
    Args:
        csv_file_list (list): List of paths to CSV files to process.
        output_dir (str): Directory to save processed CSV files.
        statements (dict): Dictionary containing category and subcategory questions.
        processor: Pretrained processor for text and image input.
        model: Pretrained model for generating labels.
    
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    for csv_file in tqdm(csv_file_list):
        # Define input and output paths
        output_csv_path = os.path.join(output_dir, f"processed_{os.path.basename(csv_file)}")

        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Initialize a dictionary to store labels for each row
        all_labels = []

        # Loop through each bounding box in the CSV file
        for index, row in df.iterrows():
            image_path = row['image_path']
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']

            # Load the image
            image = Image.open(image_path)

            # Crop the image using bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))

            labels = dict()

            # Iterate through categories and subcategories to generate labels
            for category, variations in statements.items():
                for sub_category, questions in variations.items():
                    inputs = processor(text=questions, images=cropped_image, return_tensors="pt", padding=True)
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

                    # Get the position with maximum probability
                    max_prob_pos = probs[0].argmax().item()

                    labels[sub_category] = max_prob_pos

            # Append the labels to the list for this row
            all_labels.append(labels)

        # Convert the list of labels into a DataFrame
        labels_df = pd.DataFrame(all_labels)

        # Concatenate the original DataFrame with the new labels DataFrame
        output_df = pd.concat([df, labels_df], axis=1)

        # Save the resulting DataFrame to a new CSV file
        output_df.to_csv(output_csv_path, index=False)

        print(f"Processed CSV with labels saved to {output_csv_path}")



# Example usage
csv_files = ["/home/saiteja/Desktop/lectures/data_pruning/analysis_fp/output/caption/cg.csv"]
output_directory = "output/clip_captions"

# Assuming processor and model are already initialized
process_csv_files(csv_files, output_directory, statements, processor, model)

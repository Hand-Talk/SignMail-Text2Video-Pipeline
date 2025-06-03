import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add paths to both project directories
CURRENT_DIR = Path(__file__).parent.absolute()
TEXT2SIGN_DIR = CURRENT_DIR / "AST-Avatar-SignMail-Text2Video-main"
GENERATION_DIR = CURRENT_DIR / "ailab_Generation_pipeline-dev"

sys.path.extend([str(TEXT2SIGN_DIR), str(GENERATION_DIR)])

# Import required modules from both projects
from Text2SignID_run import process_text_to_signID
import main as generation_pipeline

def setup_environment():
    """Setup necessary environment and configurations"""
    # Load environment variables (for OpenAI API key)
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Please set OPENAI_API_KEY in your .env file")

def process_text_to_video(input_text, output_video_path="./output_video.mp4", 
                         dictionary="rdp", num_interpolation=8, 
                         style_image_path=None):
    """
    Process text input to generate sign language video
    
    Args:
        input_text (str): Input text to be converted to sign language
        output_video_path (str): Path where the output video will be saved
        dictionary (str): Which sign dictionary to use ('rdp' or 'ht')
        num_interpolation (int): Number of frames to interpolate between videos
        style_image_path (str): Optional path to style image for normalization
    """
    print("Step 1: Converting text to SignIDs...")
    # Create a temporary file for the input text
    with open("temp_input.txt", "w", encoding="utf-8") as f:
        f.write(input_text)
    
    # Process text to get SignIDs
    output_csv = "temp_output.csv"
    sign_ids = process_text_to_signID("temp_input.txt", output_csv, dictionary)
    
    print("Step 2: Generating video from SignIDs...")
    # Prepare arguments for generation pipeline
    args = {
        "sign_ids": sign_ids,
        "num_interpolation": num_interpolation,
        "num_insert_interpolation": 0,  # Default value
        "style_image_path": style_image_path
    }
    
    # Run generation pipeline
    generation_pipeline.main(args)
    
    # Clean up temporary files
    os.remove("temp_input.txt")
    os.remove(output_csv)
    
    print(f"Process completed! Output video saved to: {output_video_path}")

def main():
    """Main function to run the complete pipeline"""
    setup_environment()
    
    # Example usage
    input_text = """
    Hello! This is a test message. How are you doing today?
    """
    
    try:
        process_text_to_video(
            input_text=input_text,
            output_video_path="./output_video.mp4",
            dictionary="rdp",
            num_interpolation=8,
            style_image_path=None  # Optional: provide path to style image if needed
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
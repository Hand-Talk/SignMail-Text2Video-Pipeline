import os
from dotenv import load_dotenv
from Text2Gloss_voicemail import (
    get_api_client,
    get_file_paths,
    process_initial_gloss,
    process_numbers_gloss,
    process_mm_gloss,
    splitWriteResults
)
from Gloss2SignID import GlossToSignID
import pandas as pd
import time
import sys

# Load environment variables
load_dotenv()

# Get the root directory (where the script is located)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def text_to_gloss(text: str) -> pd.DataFrame:
    """
    Convert input text to ASL gloss using Text2Gloss_voicemail pipeline
    
    Args:
        text: Input text to convert to ASL gloss
        
    Returns:
        DataFrame containing English text and ASL gloss
    """
    # Save input text to temporary file
    prompts_dir = os.path.join(ROOT_DIR, "prompts")
    temp_input_file = os.path.join(prompts_dir, "temp_input.txt")
    with open(temp_input_file, "w") as f:
        f.write(text)

    # Initialize API client
    openai_client = get_api_client()
    
    # Get file paths
    paths = get_file_paths()
    
    # Override the test sentences path with our temporary input file
    paths["test_sentences_path"] = temp_input_file
    
    # Process initial gloss
    results = process_initial_gloss(openai_client, paths, api="openai")

    results_mm = process_mm_gloss(openai_client, paths, api="openai")   
    
    # Process numbers gloss
    results_numbers = process_numbers_gloss(openai_client, paths, api="openai")
    
    # Split and write final results
    df_pairs = splitWriteResults(results_numbers, paths["output_path"], "temp_output.csv")
    
    # Clean up temporary file
    os.remove(temp_input_file)
    
    return df_pairs

def gloss_to_signid(df: pd.DataFrame, sign_type: str = "rdp") -> pd.DataFrame:
    """
    Convert ASL gloss to sign IDs using Gloss2SignID pipeline
    
    Args:
        df: DataFrame containing English text and ASL gloss
        sign_type: Type of signs to use ('ht' for HandTalk or 'rdp' for RDP signs, default: 'rdp')
        
    Returns:
        DataFrame with sign IDs added
    """
    # Initialize GlossToSignID converter
    valid_signs_path_ht = os.path.join(ROOT_DIR, "signID_dicts", "allSignIds.json")
    valid_signs_path_rdp = os.path.join(ROOT_DIR, "signID_dicts", "RDP_signs_fuller.json")
    
    # Select the appropriate signs path based on sign_type
    valid_signs_path = valid_signs_path_ht if sign_type.lower() == "ht" else valid_signs_path_rdp
    
    converter = GlossToSignID(valid_signs_path)
    
    # Process the dataframe to add sign IDs
    df_with_signids = converter.process_dataframe(df)
    
    return df_with_signids

def text_to_signid(text: str, output_file: str = "final_output.csv", sign_type: str = "rdp") -> pd.DataFrame:
    """
    Convert input text to sign IDs by running both pipelines in sequence
    
    Args:
        text: Input text to convert to sign IDs
        output_file: Name of the output file to save results (default: "final_output.csv")
        sign_type: Type of signs to use ('ht' for HandTalk or 'rdp' for RDP signs, default: 'rdp')
        
    Returns:
        DataFrame containing English text, ASL gloss, and sign IDs
    """
    start_time = time.time()
    
    print("Step 1: Converting text to ASL gloss...")
    df_gloss = text_to_gloss(text)
    
    print("\nStep 2: Converting ASL gloss to sign IDs...")
    df_final = gloss_to_signid(df_gloss, sign_type)
    # if sign_type == "rdp":
    #     df_final['id_sentence'] = df_final['id_sentence'].str.replace('¶', '')
    
    # Save final results
    output_path = os.path.join(ROOT_DIR, "output", output_file)
    df_final.to_csv(output_path, index=False)
    print(f"\nFinal results saved to: {output_path}")
    
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    
    return df_final

def main():
    """
    Main function that reads input from a text file and processes it through the pipeline.
    The input file should be provided as a command line argument, or it will use a default path.
    The output file can be provided as a second command line argument, or it will use a default name.
    The sign type can be provided as a third command line argument ('ht' or 'rdp', default: 'rdp').
    """
    
    # Default input file path
    default_input_path = os.path.join(ROOT_DIR, "input", "sample_voicemail.txt")
    default_output_file = os.path.join(ROOT_DIR, "output", "final_output.csv")
    default_sign_type = "rdp"
    
    # Get input file path from command line argument if provided, otherwise use default
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input_path
    
    # Get output file name from command line argument if provided, otherwise use default
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output_file
    
    # Get sign type from command line argument if provided, otherwise use default
    sign_type = sys.argv[3] if len(sys.argv) > 3 else default_sign_type
    
    # Validate sign type
    if sign_type.lower() not in ["ht", "rdp"]:
        print(f"Error: Invalid sign type '{sign_type}'. Must be either 'ht' or 'rdp'")
        print("Example: python Text2SignID_run.py path/to/input.txt [output_filename.csv] [ht|rdp]")
        sys.exit(1)
    
    try:
        # Read the input text file
        with open(input_file, 'r') as f:
            input_text = f.read()
            
        print(f"Reading input from: {input_file}")
        print("Input text:")
        print("-" * 50)
        print(input_text)
        print("-" * 50)
        print(f"Using SignID dictionary type: {sign_type}")
        
        # Process the text through the pipeline
        df_result = text_to_signid(input_text, output_file, sign_type)
        print("\nProcessing complete!")
        # print(df_result)
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        print("Please provide a valid input file path as a command line argument")
        print(f"Example: python Text2SignID_run.py path/to/input.txt [output_filename.csv] [ht|rdp]")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

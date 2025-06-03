import time
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Get the root directory (where the script is located)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def openReadFiles(inpath):
        inFile = open(inpath, "r")
        data = inFile.read()
        #print(data)
        inFile.close()
        return data
    
def getOpenAIResponse(system_prompt, user_message, client):
    """Get response from OpenAI API"""
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    completion = client.chat.completions.create(
        model="o1",
        messages=messages,
        temperature=1
    )
    return completion.choices[0].message.content



def getASLGloss(querys, instruct, rules, examples, client, api="openai"):
        """Get ASL gloss using specified API"""
        system_prompt = f"{instruct}\n\n{rules}\n\n{examples}"
        user_message = querys
        openAI_client = client
        
        if api.lower() == "openai":
            return getOpenAIResponse(system_prompt, user_message, openAI_client)
        else:
            raise ValueError("Invalid API specified. Choose 'openai'.")
        

def writeOut(entry, outfile):
    '''Writes out LLM results if the results are strings'''
    with open(outfile, "w") as f:
        f.write(entry + "\n\n")
        print(f"File written to {outfile}")  # Add an extra newline between entries


            
def setPromptingData(instructPath, examplesPath, rulesPath, queryPath):
        """Set up prompting data (can be used for ASL Gloss or Simple English Gloss)"""
        instruct = openReadFiles(instructPath)
        rules = openReadFiles(rulesPath)
        examples = openReadFiles(examplesPath)
        querys = openReadFiles(queryPath)
        return instruct, rules, examples, querys

def setPromptingDataMultiInstruct(instructPaths, examplesPath, rulesPath, queryPath):
        """Set up prompting data (can be used for ASL Gloss or Simple English Gloss)
        instructPaths: List of paths to instruction files that will be concatenated
        """
        # Handle multiple instruction files
        instruct = ""
        for path in instructPaths:
            instruct += openReadFiles(path)
            
        rules = openReadFiles(rulesPath)
        examples = openReadFiles(examplesPath) 
        querys = openReadFiles(queryPath)
        return instruct, rules, examples, querys

def splitWriteResults(results, output_path, filename):
    '''Splits the results into English and ASL gloss and writes them to a CSV file
    Expected format:
    English1 // ASL_GLOSS1
    English2 // ASL_GLOSS2
    ...
    
    Also handles format with extra newlines:
    English1\n // ASL_gloss1\n\n
    English2\n // ASL_gloss2\n\n
    '''
    # Create lists to store English and ASL gloss strings
    english_texts = []
    asl_glosses = []

    # Normalize the input by:
    # 1. Replacing newlines before the separator with a space
    # 2. Replacing multiple newlines with a single newline
    normalized_results = results.replace('\n //', ' //').replace('\n\n', '\n').strip()
    
    # Split results by newlines and process each line
    lines = normalized_results.split('\n')
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Split line at '//' and append to appropriate list
        parts = line.split('//')
        if len(parts) == 2:
            english_texts.append(parts[0].strip())
            asl_glosses.append(parts[1].strip())
        else:
            # Handle lines without proper separator
            print(f"Warning: Skipping malformed line: {line}")

    # Create DataFrame
    df = pd.DataFrame({
        'English': english_texts,
        'ASL Gloss': asl_glosses
    })

    # Write DataFrame to CSV
    csv_output_path = os.path.join(output_path, filename)
    df.to_csv(csv_output_path, index=False)
    print(f"Translations saved to: {csv_output_path}")
    return df
    

    
def driverFunc(querys, instruct, rules, examples, client, outfile, api="openai"):
    """Driver function for getting ASL gloss"""
    aslGloss_results = getASLGloss(querys, instruct, rules, examples, client, api)
    writeOut(aslGloss_results, outfile)
    return aslGloss_results

def get_api_client():
    """Initialize and return the API client"""
    # Try to get API key from environment variable first
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    
    return OpenAI(api_key=openai_api_key)

def get_file_paths():
    """Return a dictionary of file paths used in the pipeline"""
    prompts_dir = os.path.join(ROOT_DIR, "prompts/o1-sable")
    number_prompts_dir = os.path.join(ROOT_DIR, "prompts/o1-numbers")
    mm_prompts_dir = os.path.join(ROOT_DIR, "prompts/o1-mm")
    input_dir = os.path.join(ROOT_DIR, "input")
    output_dir = os.path.join(ROOT_DIR, "output")
    
    paths = {
        "base_path": prompts_dir,
        "output_path": output_dir,
        
        # Regular gloss paths
        "instruction_path": os.path.join(prompts_dir, "intro_instructions_combine_dev_o1_signMail.txt"),
        "rules_path": os.path.join(prompts_dir, "edited_formatting_rules_o1_signMail.txt"),
        "examples_path": os.path.join(prompts_dir, "examples_longer1_dev_o1_signMail.txt"),
        "test_sentences_path": os.path.join(input_dir, "sample_voicemail.txt"), 
        "outfile": os.path.join(output_dir, "generated_voicemail_gloss.txt"),

        # Mouth Morpheme paths
        "instruction_path_mm": os.path.join(mm_prompts_dir, "intro_instructions_combine_dev_o1_mm.txt"),
        "rules_path_mm": os.path.join(mm_prompts_dir, "edited_formatting_rules_o1_mm.txt"),
        "examples_path_mm": os.path.join(mm_prompts_dir, "examples_longer1_dev_o1_mm.txt"), 
        "outfile_mm": os.path.join(output_dir, "generated_voicemail_gloss_mm.txt"),
        "infile_mm": os.path.join(output_dir, "generated_voicemail_fixed.txt"),### ask Amber about this
        
        # Numbers model paths
        "instruction_numbers_path": os.path.join(number_prompts_dir, "intro_instructions_combine_dev_o1_numbers.txt"),
        "rules_numbers_path": os.path.join(number_prompts_dir, "edited_formatting_rules_o1_numbers.txt"),
        "examples_numbers_path": os.path.join(number_prompts_dir, "examples_numbers.txt"),
        "outfile_numbers": os.path.join(output_dir, "generated_voicemail_glossNumTest.txt"),
        "outfile_numbers_csv": os.path.join(output_dir, "generated_voicemail_glossNum.csv")
    }
    
    return paths

def process_initial_gloss(client, paths, api="openai"):
    """Process the initial ASL gloss"""
    print("Processing initial ASL gloss...")
    
    # Set variables with prompting data
    instruct, rules, examples, querys = setPromptingData(
        paths["instruction_path"], 
        paths["examples_path"], 
        paths["rules_path"], 
        paths["test_sentences_path"]
    )
    
    # Run the query
    results = driverFunc(querys, instruct, rules, examples, client, paths["outfile"], api=api)
    return results

def process_numbers_gloss(client, paths, api="openai"):
    """Process the numbers ASL gloss"""
    print("Processing numbers ASL gloss...")
    
    # The input for numbers processing is the output from the initial gloss
    # test_sentences_numbers_path = paths["outfile"]
    test_sentences_numbers_path = paths["outfile_mm"]
    
    # Set variables with prompting data
    instruct_numbers, rules_numbers, examples_numbers, querys_numbers = setPromptingData(
        paths["instruction_numbers_path"], 
        paths["examples_numbers_path"], 
        paths["rules_numbers_path"], 
        test_sentences_numbers_path
    )
    
    # Run the numbers query
    results_numbers = driverFunc(
        querys_numbers, 
        instruct_numbers, 
        rules_numbers, 
        examples_numbers, 
        client, 
        paths["outfile_numbers"], 
        api=api
    )
    
    return results_numbers

def process_mm_gloss(client, paths, api="openai"):
    """Process the mouth morphemes for ASL gloss"""
    print("Processing mouth morphemes for ASL gloss...")

    test_sentences_mm_path = paths["outfile"]

    # Set variables with prompting data
    instruct_mm, rules_mm, examples_mm, querys_mm = setPromptingData(
        paths["instruction_path_mm"], 
        paths["examples_path_mm"], 
        paths["rules_path_mm"], 
        test_sentences_mm_path,

    )
    
    results_mm = driverFunc(
        querys_mm, 
        instruct_mm, 
        rules_mm, 
        examples_mm, 
        client, 
        paths["outfile_mm"], 
        api=api
    )

    return results_mm

def main():
    """Main function to run the Text2Gloss pipeline"""
    start_time = time.time()  # Start the timer
    
    # Initialize API client
    openai_client = get_api_client()
    
    # Get file paths
    paths = get_file_paths()
    
    # Process initial gloss
    results = process_initial_gloss(openai_client, paths, api="openai")

    #Process results with mm
    results_mm = process_mm_gloss(openai_client, paths, api="openai")
    
    # Process numbers gloss
    results_numbers = process_numbers_gloss(openai_client, paths, api="openai")
    
    # Split and write final results
    df_pairs = splitWriteResults(results_numbers, paths["output_path"], paths["outfile_numbers_csv"])
    
    # End the time and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
import os
import json
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Constants
PILCROW = "¶"
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.2

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Read and return the contents of a JSON file.
    
    Args:
        file_path: Path to the JSON file.
    
    Returns:
        Dictionary containing the JSON data if successful, None otherwise.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None
    

def getSignIdDict(signs_valid: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Combine validated and non-validated sign dictionaries with pilcrow-prefixed keys.
    
    Args:
        signs_valid: Dictionary of validated signs.
        signs_invalid: Dictionary of non-validated signs.
        
    Returns:
        Combined dictionary with pilcrow-prefixed keys and cleaned values.
    """
    signIDs = {}
    
    if signs_valid:
        signIDs.update(signs_valid)

        
    # Add pilcrow to start of each signID string and clean values
    return {f"{PILCROW}{key}": [word.rstrip() for word in value] 
            for key, value in signIDs.items()}

def remove_hyphens_from_single_chars(text):
    """
    Remove hyphens from single characters in the ASL Gloss column.
    """
    # Regular expression to match single characters separated by hyphens
    pattern = r'\b(?:[A-Za-z0-9]-)+[A-Za-z0-9]\b'
    
    def replace_match(match):
        return match.group(0).replace('-', '')
    
    return re.sub(pattern, replace_match, text)


def check_remaining_hyphens(pairs_df):
    """
    Check for any remaining hyphenated single characters in the ASL Gloss column.
    """
    # Optionally, you can check for any remaining hyphenated single characters
    remaining_hyphens = pairs_df['ASL Gloss'].str.contains(r'\b(?:[A-Za-z0-9]-)+[A-Za-z0-9]\b')
    if remaining_hyphens.any():
        print("\nRows with remaining hyphenated single characters:")
        print(pairs_df[remaining_hyphens])
    else:
        print("\nAll hyphenated single characters have been processed.")
        


def cleanup_asl_gloss(text):
    """
    Remove '#' from words and trailing '+' characters in ASL Gloss.
    """
    # Function to remove '#' from words and trailing '+' characters in ASL Gloss
    words = text.split()
    processed_words = []
    for word in words:
        # Remove '#' if it's at the beginning of the word
        if word.startswith('#'):
            word = word[1:]
        # Remove '+' if it's at the end of the word
        if word.endswith('+'):
            word = word[:-1]
        processed_words.append(word)
    return ' '.join(processed_words)

def tokenize_gloss(gloss):
    """  
    Splits the gloss into separate tokens, ensuring that:
    1. Special patterns like [.*[ and ]] are kept as single tokens
    2. Other bracket symbols like [^[ and punctuation are spaced away from words
    """  
    # First handle the special [.*[ pattern
    gloss = re.sub(r'(\[.*?\[)', lambda m: f" {m.group(1)} ", gloss)
    # Handle the ]]+ pattern
    gloss = re.sub(r'(\]{2,})', r' \1 ', gloss)
    
    tokens = gloss.split()
    return tokens

def is_special_token(word):
    """
    Check if a word is a special token that should be preserved exactly.
    Special tokens are either:
    1. A token in the format [.*?[ (anything enclosed in open brackets)
    2. End tokens ']]' or larger 
    """
    # Check for ]] token
    if len(word) >= 2 and all(char =="]" for char in word):
        return True
    # Check for [.*?] pattern
    if word.startswith('[') and word.endswith('['):
        return True

    return False


def getUniqueASLGlossWords(pairs_df):
    # Create a dictionary to store unique words from ASL Gloss
    asl_gloss_words = {}

    # Iterate through each row in the DataFrame
    for gloss in pairs_df['ASL Gloss']:
        # Tokenize the gloss, convert to lowercase, and strip punctuation
        tkn = tokenize_gloss(gloss)
        tknString = " ".join(tkn)
        words = [word.lower().strip('+.,!?:;()[]^{}') for word in tknString.split() if is_special_token(word) == False]
        
        # Add each word to the dictionary
        for word in words:
            if word not in asl_gloss_words:
                asl_gloss_words[word] = True

    # Print the number of unique words
    print(f"Number of unique words in ASL Gloss: {len(asl_gloss_words)}")
        
    return asl_gloss_words

def add_signIDs_to_gloss_words(asl_gloss_words, signIDs):
    """
    Add signIDs and their associated glosses to the asl_gloss_words dictionary.
    
    Args:
        asl_gloss_words (dict): Dictionary mapping gloss words to True
        signIDs (dict): Dictionary mapping signIDs to lists of associated glosses
        
    Returns:
        dict: Updated asl_gloss_words with signIDs and glosses
    """
    for word in asl_gloss_words:
        # If word contains hyphen, create a space-separated version
        lookup_word = word.replace('-', ' ').lower()
        lookup_word_list = lookup_word.split()
        lookup_length = len(lookup_word_list)
        if lookup_length == 3: 
            lookup_word = lookup_word_list[1]
        
        # Search for the word in the values of signs_valid
        for sign_id, glosses in signIDs.items():
            # Convert glosses to lowercase for comparison
            lower_glosses = [gloss.lower() for gloss in glosses]
            
            # Check if our lookup word is in the glosses, either as an exact match
            # or as part of a longer phrase
            for gloss in lower_glosses:
                if lookup_word == gloss or f" {lookup_word} " in f" {gloss} ":
                    # Convert spaces to hyphens in the glosses for the new entry
                    hyphenated_glosses = [gloss.replace(' ', '-') for gloss in glosses]
                    new_entry = {sign_id: hyphenated_glosses}
                    
                    if isinstance(asl_gloss_words[word], dict):
                        if asl_gloss_words[word] != new_entry:
                            asl_gloss_words[word] = [asl_gloss_words[word], new_entry]
                    elif isinstance(asl_gloss_words[word], list):
                        if new_entry not in asl_gloss_words[word]:
                            asl_gloss_words[word].append(new_entry)
                    else:
                        asl_gloss_words[word] = [new_entry]
                    break  # Found a match, no need to check other glosses for this signID
    
    return asl_gloss_words




def get_word_senses(gloss_word, word_senses_dict):
  """
  Retrieve the word senses for a given gloss word.
  """
  gloss_word = gloss_word.rstrip('.,!?:;+')
  return word_senses_dict.get(gloss_word.lower(), [])


def format_word_senses(word_senses):
  """
  Format the word senses for inclusion in the prompt.
  """
  formatted = []
  for sense in word_senses:
      for id, words in sense.items():
          formatted.append(f"ID: {id}, Words: {', '.join(words)}")
  return '\n'.join(formatted)


def get_word_senses_for_gloss(gloss, word_senses_dict):  
    """
    Get word senses for all tokens in the gloss.
    """
    tokens = tokenize_gloss(gloss)
    all_senses = {}
    
    for token in tokens:
        # Skip special tokens
        if is_special_token(token):
            continue
            
        # Clean and lowercase the token for lookup
        cleaned_token = token.lower().strip('.,!?:;+')
        
        # Try to get word senses
        senses = word_senses_dict.get(cleaned_token, None)
        if senses and not isinstance(senses, bool):
            all_senses[token] = senses
            
    return all_senses



def get_word_sense_ids(english, gloss, word_senses_dict):
  """
  Use the LLM to determine the correct word sense for each word in the gloss.
  """
  gloss_tokens = tokenize_gloss(gloss)
  all_senses = get_word_senses_for_gloss(gloss, word_senses_dict)

  prompt = f"""
Given the following:

English sentence: {english}
ASL Gloss: {gloss}

Your task is to replace each gloss token with either:
1. The most appropriate sense ID if available  (THIS IS YOUR PRIMARY TASK)
2. The EXACT original token if it is a special token or has no sense ID

CRITICAL: You must preserve special tokens (like [^[, [mm:1[, ]], etc.) EXACTLY as they appear.
If you see consecutive special tokens (like ]] ]]), keep them EXACTLY as they are.
DO NOT modify, combine, or split any special tokens - copy them exactly as they appear.

If no sense ID is available for a particular token, keep it exactly.  
Preserve all special tokens in the final output EXACTLY as they appear,  
but ensure each token with a sense ID is replaced by the ID (do not append the gloss word).

For words with multiple possible senses, choose the one that best fits the context of the sentence.
For words without word senses, just output the original word.
Word senses for each word:
"""
  for token in gloss_tokens:
        prompt += f"\n{token}:\n" 
        if is_special_token(token):
            prompt += f"Preserve this token EXACTLY: {token}"
        elif token in all_senses:
            prompt += format_word_senses(all_senses[token])
        else:
            prompt += f"Use the original word: {token}"

         
  prompt += "\n\nRespond with a space-separated list of IDs , original words, or special tokens."
  prompt += "\nFor words with word senses, use ONLY the ID (e.g. ¶xGf7jvuSHvaHxq3hd5JH or ¶THANK-YOUasym)."
  prompt += "\nFor words without word senses or special tokens, use the original word exactly."
  prompt += "\nPreserves special tokens exactly as shown - do not modify them."
  prompt += "\nBefore submitting your response, verify that you have followed these instructions."

  messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in linguistics and sign language"},
            {"role": "user", "content": prompt}
        ]
  completion = openai_client.chat.completions.create(
      model=MODEL_NAME,
      messages=messages,
      temperature=TEMPERATURE
  )


  selected_ids = completion.choices[0].message.content.strip().split()

# Ensure the response matches the length of the gloss
  if len(selected_ids) != len(gloss_tokens):
    # raise ValueError(f"LLM response length ({len(selected_ids)}) does not match gloss length ({len(gloss_words)})")
    print(f"Warning: LLM response length ({len(selected_ids)}) does not match gloss length ({len(gloss_tokens)})")
    # Continue running despite the mismatch     

  return " ".join(selected_ids)



def process_sentence_pairs(sentence_pairs, word_senses_dict):
  """
  Process multiple English-ASL sentence pairs and return their predicted ID number sentences.
  
  :param sentence_pairs: List of tuples, each containing (English sentence, ASL Gloss sentence)
  :param word_senses_dict: Dictionary containing word senses for ASL Gloss words
  :return: List of dictionaries, each containing original sentences and predicted ID sentences
  """
  from tqdm import tqdm
  
  results = []
  unmatched = []
  
  # Create progress bar
  pbar = tqdm(total=len(sentence_pairs), desc="Processing sentence pairs")
  
  for english, asl_gloss in sentence_pairs:
      try:
          id_sentence = get_word_sense_ids(english, asl_gloss, word_senses_dict)
          results.append({
                "english": english,
                "asl_gloss": asl_gloss,
                "id_sentence": id_sentence
            })
 
      except Exception as e:
          print(f"Error processing sentence pair: ({english}, {asl_gloss})")
          print(f"Error message: {str(e)}")
      
      pbar.update(1)
      
  pbar.close()
  
  return results, unmatched




class GlossToSignID:
    """Handles conversion of ASL Gloss text to SignID sequences."""
    
    def __init__(self, valid_signs_path: str):
        """Initialize the converter with sign dictionaries.
        
        Args:
            valid_signs_path: Path to validated signs JSON file.
            invalid_signs_path: Path to non-validated signs JSON file.
        """
        self.signs_valid = read_json_file(valid_signs_path)
        self.signIDs = getSignIdDict(self.signs_valid)
        self.word_senses_dict = {}
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a DataFrame containing English and ASL Gloss columns.
        
        Args:
            df: DataFrame with 'English' and 'ASL Gloss' columns.
            
        Returns:
            DataFrame with added 'id_sentence' column.
        """
        # Rename columns if needed
        if df.columns[0] != 'English' or df.columns[1] != 'ASL Gloss':
            df = df.rename(columns={df.columns[0]: 'English', df.columns[1]: 'ASL Gloss'})
        
        # Clean up the gloss
        # df['ASL Gloss'] = df['ASL Gloss'].apply(remove_hyphens_from_single_chars)
        check_remaining_hyphens(df)
        df['ASL Gloss'] = df['ASL Gloss'].apply(cleanup_asl_gloss)
        
        # Get word senses
        asl_gloss_words = getUniqueASLGlossWords(df)
        self.word_senses_dict = add_signIDs_to_gloss_words(asl_gloss_words, self.signIDs)
        
        # Process sentences
        sentence_pairs = list(zip(df['English'], df['ASL Gloss']))
        results, unmatched = process_sentence_pairs(sentence_pairs, self.word_senses_dict)
        
        return pd.DataFrame(results if results else unmatched)

def main():
    """Main entry point for the script."""
    # Load configuration
    VALID = os.getenv("VALID_SIGNS_PATH", "signID_dicts/RDP_signs_fuller.json")
    INPAIRS = os.getenv("INPUT_PAIRS_PATH", "output/temp_output.csv")
    OUTFILE = os.getenv("OUTPUT_PATH", "output/final_output.csv")
    
    # Initialize converter
    converter = GlossToSignID(VALID)
    
    # Process input file
    input_df = pd.read_csv(INPAIRS)
    results_df = converter.process_dataframe(input_df)
    
    # Save results
    results_df.to_csv(OUTFILE, index=False)
    print(f"Results saved to {OUTFILE}")

if __name__ == "__main__":
    main()


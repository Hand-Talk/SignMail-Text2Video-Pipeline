# Text2SignID Pipeline

This is the AI Lab's AST Text2SignID pipeline, consisting of the first relase of o1-Sable, our full grammatical mode.  This model applies LLMs (Open AI's GPT o1 and 4o models) to convert English text (voicemails) to ASL (American Sign Language) signID sequences, which can be used with ASL dictionaries for sign selection for ASL generation.  This model makes use of a two-step process:
1. Converting English text to ASL gloss notation
2. Converting ASL gloss to corresponding signIDs


## Setup
Make sure to add your OpenAI API key
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key to the `.env` file

## Main Components

`Text2SignID_run.py` is the main script that orchestrates the entire pipeline. It provides functions to:
- Convert text to ASL gloss
- Convert ASL gloss to sign IDs
- Process input files through the complete pipeline
- Outputs a csv file containing English, ASL Gloss, and SignID sequence for ASL Generation

Usage:

To run the model, run
```bash
python Text2SignID_run.py [path/to/input.txt] [output_fileName.csv] [preferred_dictionary - `rdp` or `ht`]
```

If no input file is provided, it uses a default test file.
An example file can be found in `input/sample_voicemail.txt`.
Sample output files can also be found in `output/` folder.
If no dictionary is specified, we use `rdp` as the default.  

### `Text2Gloss_voicemail.py`
Handles the conversion of English text to ASL gloss notation using OpenAI's API. Key features:
- Processes input text through specialized prompts
  - First, handles grammatical modeling and translation
    - This first pass also predicts and adds eyebrow movement notation
  - Next, handles mouth morphemes - predicts and adds mouth morpheme notation
  - Finally, checks and corrects numbers formatting
- Outputs results as a structured format (CSV) containing the original English sentence and the ASL gloss translation

### `Gloss2SignID.py`
Converts ASL gloss notation to corresponding signIDs for either the Hand Talk dictionary or Omnibirdge's RDP dictionary. 
Contains the following:
- Makes use of both the original English sentence and the ASL gloss translation
- Sets dictionary we are working with based on user input (HT or RDP)
- Maps ASL gloss terms to signIDs
    - Only collects signIDs for words in the ASL gloss
- Provides English, ASL gloss, and all potential SignIDs for each gloss word to LLM for word sense disambiguation
- Handles invalid or missing signs
- Preserves any bracket notation indicating non-manual markers (e.g. `I [mm9[WONDER]] [^[YOU KNOW DIGITAL CAMERA [v[FIND WHERE]]]? REASON M-O-H-A-N HIS DOOR FEW PICTURE TAKE, PERSONAL USE`)
- Returns a sequence of SignIDs corresponding to each gloss sequence
- Processes input in DataFrame format

## Directory Structure

- `input/`: Contains input text files to be processed
- `output/`: Stores the generated output files
- `prompts/`: Contains prompt templates for the text-to-gloss conversion
  - `prompts/o1-sable`: Contains the prompt templates for o1-Sable, our first release (PoC)
  - `prompts/o1-mm`: Contains the prompt templates for the mouth morphemes model (which generates and applies mouth morphemes)
  - `prompts/o1-numbers`: Contains the prompt templates for the numbers model (which handles numbers formatting)
- `signID_dicts/`: Contains JSON dictionaries for sign ID mapping
  - `allSignIDs.json`: A combined json file of all sign mappings for HT dataset (currently in place)
  - `RDP_signs_fuller.json`: A combination json file of all sign mappings for RDP dataset (currently in place)
    - We may change this in the future and use an updated version that combines RDP and HT
  - Depreciated (no longer in use):
    - `signs_validated.json`: Validated sign mappings
    - `signs_not_validated.json`: Sign mappings which have not yet been validated
  

## Output Format

The pipeline generates a CSV file containing:
- Original English text
- ASL gloss notation
- Corresponding signID sequence

The signID sequence can then be used in conjunction with the ASL dictionary to locate the correct signs for each sentence.

## Dependencies

- pandas
- OpenAI API client
- Python 3.x
- dotenv
- re
- time
- json

## Error Handling

The pipeline includes error handling for:
- Missing input files
- API errors
- Invalid sign mappings

## Notes

- As of right now this pipeline is designed to handle a single voicemail .txt file.  This can be extended if needed.
- Current SignIDs now default to using Omnibridge's RDP sign dictionary, as their dictionary is more robust and has cleaner versions of the sign videos.  If users want to use Hand Talk's dictionary (now a combination of validated and unvalidated signsIDs), that option is available.  Users can specify which dictionary they want to use with an argument when calling the `run` function (see above for details). We hope that with future versions, we will be able to use a dictionary that combines both RDP and HT dictionaries for best coverage.  
- Processing time may vary depending on the length of input text and API response times.  Average time for a short voicemail (~ 4 sentences) is 1 minute 30 seconds.

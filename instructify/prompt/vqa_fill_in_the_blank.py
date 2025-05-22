import random
import re
from prompt.base_prompts import multi_turn_parsing, ParseOutputError

PROMPT = """Design a conversation between someone creating fill-in-the-blank statements and an assistant answering with the word/words.
- First, write a sentence of explaining the expected answer format (e.g. single word, phrases, xor rewriting the statement with the missing word/words)
- Base statements and answers only on given details.
- Don't invent, assume, or extrapolate.
- Asker asks directly about the image, assuming they both can see it.
- Fix grammar and spelling that the raw data may contain.

The conversation should be <CONVERSATION_LENGTH> fill-in-the-blank statements (containing <fill-in-the-blank>, where the blank(s) are), requiring answers targeting specific details. These statments should show reasoning (such as follow up statments for clarification or additional details). 

Format the conversation as follows:
Instruction (only once)
<INSTRUCTION_BREAK>
Statement: [Statement with <fill-in-the-blank>]
Answer: [Answer]"""
max_conversation_length = 5

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    conversation_length = random.randint(2, max_conversation_length)
    prompt_formatted = PROMPT.replace("<CONVERSATION_LENGTH>", str(conversation_length))
    return prompt_formatted, {"conversation_length": conversation_length}

def parse_input(input_info: list, metadata: dict):
    """
    Expects a list of strings representing the information of the image
    """
    return "\n".join(input_info)

def parse_output(input_string: str, metadata: dict):
    """
    Parse the input string for questions and answers, ignoring 'Turn X' markers.

    Args:
        input_string (str): The input string containing questions and answers, which may be in plain or bolded format.
        metadata (dict): A dictionary containing conversation metadata, such as expected conversation length.

    Returns:
        list: A list of tuples with each tuple containing a question and its corresponding answer.

    Raises:
        ParseOutputError: If no questions or answers are found, or if the number of questions doesn't match metadata.
    """
    instruction = input_string.split("<INSTRUCTION_BREAK>")[0].replace("**", "").strip()
    if "Instruction:" == instruction[:12]:
        instruction = instruction.split("Instruction:")[1].strip()
    elif "[instruction]" == instruction[:13]:
        instruction = instruction.split("[instruction]")[1].strip()

    input_string = input_string.split("<INSTRUCTION_BREAK>")[1]
    turns = multi_turn_parsing(input_string, metadata["conversation_length"], reversed=False, question_string="Statement")
    
     # remove questions that don't have "_____" in them
    filtered_turns = []
    for i in range(0, len(turns), 2):
        if "<fill-in-the-blank>" in turns[i]:
            filtered_turns.extend(turns[i:i+2])
    if len(filtered_turns) == 0:
        raise ParseOutputError("No questions found in the input string.")

    # prepend the instruction to the first turn
    if instruction:
        filtered_turns[0] = f"{instruction}\n{filtered_turns[0]}"
    return filtered_turns
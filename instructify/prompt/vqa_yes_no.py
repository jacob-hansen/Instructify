import random
import re
from prompt.base_prompts import multi_turn_parsing, ParseOutputError

PROMPT = """You're tasked with designing a conversation between someone asking yes/no questions about an image and an assistant answering Yes or No. You're given information, but first need to think of clever questions that are incorrect. Your goal is to:

- Base questions and answers only on given details.
- Asker asks directly about the image, assuming they both can see it.
- Fix grammar and spelling that the raw data may contain.
First, write a short paragraph detailing some example incorrect yes/no questions (4-6 examples). Good examples include questions that are plausible, require reasoning/goal-oriented, but are clearly denied by the information. You know anything that is mentioned in the information must be visible, so you'll have to think carefully. 
After writing this paragraph, conclude with "<FINISHED_THINKING>" Then the conversation should be <CONVERSATION_LENGTH> questions, starting with a question that also specifies that the answers should be in yes/no format. <ANSWER_ORDER>. Remember, you are putting questions, not statements, and the first question should give instruction how the answer should be formatted (e.g. Yes/No). 

Structure your response as: 
[discussion of difficult negative examples] 
<FINISHED_THINKING> 
Question: [Instruction & Question] 
Answer: [Answer]"""

def order_generator(n):
    return f"The answer order should be {', '.join(['Yes' if i % 2 == 0 else 'No' for i in range(n)])}. "

max_conversation_length = 5

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    conversation_length = random.randint(2, max_conversation_length)
    additional_info = order_generator(conversation_length)
    prompt_formatted = PROMPT.replace("<CONVERSATION_LENGTH>", str(conversation_length)).replace("<ANSWER_ORDER>", additional_info)
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
    input_string = input_string.strip().split("<FINISHED_THINKING>")[1]
    turns = multi_turn_parsing(input_string, metadata["conversation_length"])
    
    # make sure all the answers are Yes or No
    turns_filtered = []
    for i in range(0, len(turns), 2):
        if turns[i+1].strip().lower() == 'yes' or turns[i+1].strip().lower() == 'no':
            turns_filtered.extend(turns[i:i+2])
    if len(turns_filtered) == 0:
        raise ParseOutputError("No questions or answers found.")
    # assert the first question has "yes" and "no" in it (options)
    if "yes" not in turns[0].lower() or "no" not in turns[0].lower():
        turns[0] += " Answer with Yes or No."
    
    # shuffle the turns (2 at a time, skipping first pair)
    remaining_grouped_turns = [turns[i:i+2] for i in range(2, len(turns), 2)]
    random.shuffle(remaining_grouped_turns)
    flat_turns = [item for sublist in remaining_grouped_turns for item in sublist]
    base_turns = turns[:2]
    return base_turns + flat_turns
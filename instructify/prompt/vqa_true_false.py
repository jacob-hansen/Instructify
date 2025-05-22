import random
import re
from prompt.base_prompts import multi_turn_parsing, ParseOutputError

PROMPT = """You're tasked with designing a conversation between someone giving statements about an image and an assistant answering True or False. 
You're given information, but first need to think of clever statements that are incorrect. You're goal is to:
- Base statements and answers only on given details.
- Asker asks directly about the image, assuming they both can see it.
- Fix grammar and spelling that the raw data may contain.

First, write a paragraph detailing some example False statements (4-6 examples). Good examples include things that are plausible, require reasoning/goal oriented, but clearly denied by the information. You know anything that is mentioned in the information must be visible, so you'll have to think carefully. After writing this paragraph, conclude with "<FINISHED_THINKING>"
Then the conversation should be <CONVERSATION_LENGTH> statements, starting with instruction outlining that the answers should be true or false. <ANSWER_ORDER>. Remember, you are putting statements, not questions and the first statement should give instruction how the answer should be formatted (e.g. True/False). Structure your response as:
[discussion of difficult negative examples]
<FINISHED_THINKING>
Statement: [Instruction & Statement]
Answer: [Answer]"""

def order_generator(n):
    return f"The answer order should be {', '.join(['True' if i % 2 == 0 else 'False' for i in range(n)])}. "

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
    turns = multi_turn_parsing(input_string, metadata["conversation_length"], question_string="Statement")
    
    # make sure all the answers are True or False
    turns_filtered = []
    for i in range(0, len(turns), 2):
        if turns[i+1].strip().lower() == 'true' or turns[i+1].strip().lower() == 'false':
            turns_filtered.extend(turns[i:i+2])
    if len(turns_filtered) == 0:
        raise ParseOutputError("No questions or answers found.")
    # assert the first question has "true" and "false" in it (options)
    if "true" not in turns[0].lower() or "false" not in turns[0].lower():
        turns[0] += " Answer with True or False."
    
    remaining_grouped_turns = [turns[i:i+2] for i in range(2, len(turns), 2)]
    random.shuffle(remaining_grouped_turns)
    flat_turns = [item for sublist in remaining_grouped_turns for item in sublist]
    base_turns = turns[:2]
    return base_turns + flat_turns
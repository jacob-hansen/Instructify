import random
import re
from prompt.base_prompts import multi_turn_parsing

PROMPT = """Design a conversation between someone asking about an image and an assistant answering.
- Base questions and answers only on given details.
- Don't invent, assume, or extrapolate.
- Asker asks directly about the image, assuming they both can see it.
- Fix grammar and spelling that the raw data may contain.
- <CONVERSATION_FORMAT>

Note:
- Think carefully before generating each answer, especially when interpreting subtle or complex details.
- Answers should aim to reach a goal or clarify specific information, the person asking the questions has an adgenda, what they want to solve or know.
- Format the conversation to have <CONVERSATION_LENGTH> turns.

Format the conversation as follows (where instruction in the first question tells the assistant what format to provide the answer in):
Question: [Instruction] [Question]
Answer: [Answer]"""

conversation_formats = [
    "The first question gives instructions that each of answers should be a single sentence.",
    "The first question gives instructions that each of answers should be a short phrase.",
    "The conversation is succinct and to the point, with each question and answer being a single sentence."
]

max_conversation_length = 5
#focusing on understanding the relationships and interactions between objects in the scene as a whole. Guide the conversation to discuss how objects are positioned relative to each other and how they interact or influence each other. The discussion should explore how these interactions contribute to the overall context and meaning of the scene

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    conversation_length = random.randint(2, max_conversation_length)
    conversation_format = random.choice(conversation_formats)
    prompt_formatted = PROMPT.replace("<CONVERSATION_LENGTH>", str(conversation_length)).replace("<CONVERSATION_FORMAT>", conversation_format)
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
    return multi_turn_parsing(input_string, metadata["conversation_length"])
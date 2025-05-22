from prompt.base_prompts import multi_turn_parsing

PROMPT = """Design a question and answer which requires visual reasoning about what is going on in the scene, making it clear what the scene is about. 

Importantly, the question must be pithy and vague (e.g. "Describe to me ...", "In what ways ..."), requiring substantial knowledge of the scene while never asking multiple things at once. On the other hand, the answer may be detailed and provide the context needed to caption the scene in detail.

Format the question/answer pair as follows:
Question: [Question]
Answer: [Answer]"""

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    return PROMPT, {}

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
    return multi_turn_parsing(input_string, 1)
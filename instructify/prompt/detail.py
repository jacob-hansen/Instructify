from prompt.base_prompts import multi_turn_parsing

PROMPT = """Design a question and answer which requires focusing on detailed descriptions, with particular attention to describe one element of the scene. The question should be simple and concise (e.g., "Tell me about ...", "Describe in detail ..."), requiring an understanding of a single aspect of the scene without overwhelming the user by asking about multiple components at once.
All information must be based on the data provided. Make no assumptions about the scene that are not supported by the data, even if it means the answer is not long.

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
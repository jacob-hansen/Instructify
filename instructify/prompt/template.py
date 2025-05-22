PROMPT = """You are a helpful AI assistant."""

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    return PROMPT, {}

def parse_input(input_info, metadata: dict):
    """
    Parses the input and returns the expected output.

    Args:
        input_string (Unkown): The input data to parse.
        metadata (dict): Additional information to use in parsing.

    Returns:
        Any: The parsed output.
    """
    return input_info

def parse_output(input_string: str, metadata: dict):
    """
    Parses the output string and returns the expected output.

    Args:
        input_string (str): The output string to parse.
        metadata (dict): Additional information to use in parsing.

    Returns:
        Any: The parsed output.
    """
    return input_string
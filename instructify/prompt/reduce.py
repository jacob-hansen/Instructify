import re

PROMPT = """You are given a list of strings labeled 'AB', 'AC', etc. representing information about an image. Following this, you are given a conversation/questions about the same image. 
Your task is to identify the pieces of information that are referenced in the questions. Carefully choose which strings are specifically relevant. Respond with a python list of the letters associated with the information used e.g. ["AA", "AA", ...]. Provide nothing but this list."""

CHARACTERS = [chr(i) + (chr(j) if i > 64 else '') for i in range(65, 91) for j in range(65, 91) if i <= j]
CHARACTERS_TO_INDEX = {char: i for i, char in enumerate(CHARACTERS)}

def parse_input(input_info: list, metadata: dict):
    """
    Expects a dictionary containing:
        "information": list of strings representing the information of the image
        "conversation": list of strings representing the conversation about the image

    Returns:
        str: The input string. Enumerating the information as 'A', 'B', 'C', ... 'AB', 'AC', ... 'ABC', ..., and the conversation as 'Turn 1', 'Turn 2', ...
    """
    info, conv = input_info["information"], input_info["conversation"]
    if type(conv) == list:
        conv = "\n".join(conv)
    info_dict = {CHARACTERS[i]: info[i] for i in range(len(info))}
    input_string = "\n".join([f"{k}: {v}" for k, v in info_dict.items()]) + "\n\nConversation:\n" + conv
    return input_string
    

def parse_output(input_string: str, metadata: dict):
    """
    Parse the input string for a list of strings.

    Args:
        input_string (str): The input string containing a list of strings.
        metadata (dict): A dictionary containing conversation metadata (not used in this implementation).

    Returns:
        list: A list of strings extracted from the input.

    Raises:
        Exception: If no information is found.
    """
    match = re.search(r'\[.*?\]', input_string)
    if not match:
        raise Exception("No information found")
    
    list_content = match.group(0)
    info = re.findall(r'[A-Z]+', list_content)
    return [CHARACTERS_TO_INDEX[char] for char in info]
    
    
    

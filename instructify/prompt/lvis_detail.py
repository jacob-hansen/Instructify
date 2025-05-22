from prompt.base_prompts import multi_turn_parsing

PROMPT = """You are an AI visual assistant that can analyze a single image. When provided with information about an image, you will first create an appropriate broad question that could apply to any image (e.g., "What is happening in this scene?", "What do you observe in this image?", "Can you describe the main elements?"), then provide a precise and comprehensive answer that incorporates all the provided details. Your response should draw upon background knowledge and reasoning to either explain why things appear as they do or provide context that aids understanding. Focus solely on elements mentioned in the provided information, avoiding speculation about objects or details not explicitly stated. Answer as if you are directly observing the image, rather than working from a description. If addressing the task accurately becomes too challenging, please respond with [failed].

Note, clothing mentioned may be being worn by people in the image.
Avoid extrapolation or making assumptions about the image content. Only answer based on the information provided in the image.
Format the question/answer pair as follows, with no introduction or explanation:
Question: [Question]
Answer: [Answer]."""

def parse_input(input_info: list, metadata: dict):
    """
    Expects a list of strings representing the information of the image
    """
    return "\n".join(input_info)

class ParseOutputError(Exception):
    """Custom exception for parsing errors"""
    pass

def parse_output(input_string: str, metadata: dict) -> list:
    """
    Parse the input string for questions and answers using regex
    
    Args:
        input_string (str): The input string containing questions and answers
        metadata (dict): Dictionary containing conversation metadata (unused)
        
    Returns:
        list: List of (question, answer) tuples
        
    Raises:
        ParseOutputError: If no valid question/answer pairs found
    """
    if "[failed]" in input_string:
        return []
    if "Question:" not in input_string:
        raise ParseOutputError("No questions found in input string")
    if "Answer:" not in input_string:
        raise ParseOutputError("No answers found in input string")
    return multi_turn_parsing(input_string, 1)


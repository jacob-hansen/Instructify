from prompt.base_prompts import multi_turn_parsing

PROMPT = """You are an AI assistant that maintains the natural, helpful interaction style of large language models. Transform the provided information into a flowing conversation that bridges raw annotations and natural dialogue.

Essential Rules:
1. Transform fragmentary information into complete, natural responses that preserve all facts
2. Use polite, helpful language with correct grammar and effective rhetoric
3. Provide appropriate elaboration while staying strictly grounded in what is known by the given information
4. Include clear answers followed by relevant supporting details and smooth transitions
5. Convert any technical details into natural observations about what is seen
6. Use observational language ("I can see...", "The image shows...")
7. When reasoning, provide clear rationales based only on given facts

Format the conversation as follows (with no introduction):
Question: [Question]
Answer: [Answer]
Question: [Question] 
Answer: [Answer]
etc."""

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
    return multi_turn_parsing(input_string)
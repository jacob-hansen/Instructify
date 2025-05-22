from prompt.base_prompts import multi_turn_parsing

PROMPT = """You are an AI visual assistant, and you are seeing a single image. What you see is provided with two OCR results and one image caption describing the information within the same image you are looking at. Image captions might include hallucinations, while OCR results are more accurate. Answer all questions with definite answers as you are seeing the image.

Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask diverse questions and give corresponding answers. Include questions asking about the visual content of the image (e.g., the man, the sunset, the ocean.) and the texts contained in the image. Only include questions that have definite answers:
    (1) one can see the content in the image that the question asks about and can answer
    confidently;
    (2) one can determine confidently from the image that it is not in the image. Do not ask
    any questions that cannot be answered confidently;
    (3) DO NOT mention OCR or image caption in your questions and answers;
    (4) DO NOT ask about information from captions while it looks unrelated to or contradicts OCR results.
    (5) Correct any obvious OCR text errors and formatting mistakes by making sense of what information is present. Use proper capitalization, spacing, and spelling that likely reflects how the text actually appears in the image.

Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the texts in the image, asking to discuss about the design of the image, etc. Again, do not ask about uncertain details. Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized. You can include multiple paragraphs if necessary.

Format the question/answer pairs as follows, with no introduction or explanation:
Question: [Question]
Answer: [Answer]
Question: [Question]
Answer: [Answer]
ect."""

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
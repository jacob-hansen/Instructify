from prompt.base_prompts import multi_turn_parsing

PROMPT = """You are an AI visual assistant, and you are seeing a single image. Answer all questions as you are seeing the image. Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask diverse questions and give corresponding answers. Include questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers: (1) one can see the content in the image that the question asks about and can answer confidently; (2) one can determine confidently from the image that it is not in the image. Do not ask any questions that cannot be answered confidently. Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss events happening in the image, etc. Again, do not ask about uncertain details, but the questions should be challenging enough, requiring the person to utilize 1) complex reasoning; 2) world knowledge; 3) explanatory answers; and 4) multi-turn conversation, to give accurate answers. Please provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized. Please ensure all the questions are closely related to the visual content of the provided image, which means that if the person cannot see the picture but only gets access to the text description of it, he/she will not be able to answer accurately. If the AI assistant asks counterfactual questions, the person should give a negative answer, rather than making up an answer.

Note, clothing mentioned may be being worn by people in the image. Do not assume things.
Avoid extrapolation or making assumptions about the image content. Only answer based on the information provided in the image.
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
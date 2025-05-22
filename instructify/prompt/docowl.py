from prompt.base_prompts import multi_turn_parsing

PROMPT = """You are an AI document analysis assistant examining a text-rich image through provided descriptions. Respond as if directly viewing the image.
Create a Q&A dialogue that demonstrates understanding of:
- Core content and organization
- Visual structure and elements
- Relationships between components
- Quantitative aspects where relevant

Questions should:
- Be clearly answerable from the content
- Span multiple complexity levels
- Include synthesis tasks
- Focus on connections between elements

Answers should:
- Show step-by-step reasoning when applicable
- Paraphrase rather than quote
- Break down complex analysis
- Use natural language

Critical Instruction: 
- The dialogue length should match the depth and complexity of the provided content, without fabricating additional details. Sometimes this means the dialogue will be short.
- To avoid errors that occur in ocr, never directly quote text from the document. Instead, paraphrase and describe content in your own words.

Format the question/answer pairs as follows, with no introduction or explanation:
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
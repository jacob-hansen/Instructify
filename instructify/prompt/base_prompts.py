import re

class ParseOutputError(Exception):
    """Custom exception for parsing errors."""
    pass

multi_turn_conversation = """Design a conversation between someone asking about an image and an assistant answering.
- Base questions and answers only on given details.
- Don't invent, assume, or extrapolate.
- Asker asks directly about the image, assuming they both can see it.

<PROMPT>

Format the conversation as follows:
Question: [Question]
Answer: [Answer]"""

def multi_turn_parsing(input_string: str, num_conversation_steps: int = -1, reversed: bool = False, question_string: str = "Question"):
    """
    Parse the input string for questions and answers, ignoring 'Turn X' markers.

    Args:
        input_string (str): The input string containing questions and answers, which may be in plain or bolded format.
        num_conversation_steps (int): The number of conversation steps expected in the input string.

    Returns:
        list: A list of tuples with each tuple containing a question and its corresponding answer.

    Raises:
        ParseOutputError: If no questions or answers are found, or if the number of questions doesn't match metadata.
    """
    # Remove any lines with 'Turn X', where X is a number. Max length of the line is 10 characters.
    input_string = re.sub(r"Turn \d{1,2}\n", "", input_string)

    # remove everything before either the first question_string (or "Answer" if reveresed)
    if not reversed:
        input_string = re.sub(rf".*?{question_string}", question_string, input_string, count=1, flags=re.DOTALL)
    else:
        input_string = re.sub(rf".*?Answer", "Answer", input_string, count=1, flags=re.DOTALL)

    # Regex to find questions and answers (normal or bolded)
    if not reversed:
        qa_pattern = re.compile(rf"{question_string}:\s*(.*?)\nAnswer:\s*(.*?)(?=\n\n{question_string}:|$)", re.DOTALL)
    else:
        qa_pattern = re.compile(r"Answer:\s*(.*?)\n{question_string}:\s*(.*?)(?=\n\nAnswer:|$)", re.DOTALL)

    # Find all Q&A pairs
    matches = qa_pattern.findall(input_string)
    
    if not matches:
        raise ParseOutputError("No questions or answers found in the input string.")
    
    # Validate conversation length
    if num_conversation_steps != -1:
        if num_conversation_steps is not None and len(matches) != num_conversation_steps:
            raise ParseOutputError(f"Conversation length mismatch. Expected {num_conversation_steps} questions, found {len(matches)}.")
    elif len(matches) == 0:
        raise ParseOutputError("Invalid number of questions and answers.")
    
    # Return Q&A pairs as flat list
    conv = []
    for match in matches:
        match_0 = match[0].strip()
        match_1 = match[1].strip()
        if not reversed:
            if len(match_1.split()) < 3 and match_1[-1] == ".":
                match_1 = match_1[:-1]
            conv.extend([match_0, match_1])
        else:
            if len(match_0.split()) < 3 and match_0[-1] == ".":
                match_0 = match_0[:-1]
            conv.extend([match_1, match_0])
    return conv
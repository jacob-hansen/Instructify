PROMPT = """Convert the following to statement(s) of fact, including any negatives, if applicable. Directly provide the sentences, separated by line breaks, with no explanation, fixing any gramatical errors or weird wording. Avoid repetition. All statements should be declarative."""

def parse_input(input_info, metadata: dict):
    """
    Parses the input and returns the expected output.

    Args:
        input_string (Unkown): The input data to parse.
        metadata (dict): Additional information to use in parsing.

    Returns:
        Any: The parsed output.
    """
    QA = []
    for dataset in input_info:
        QA.extend(input_info[dataset]["QA"])
    return "\n".join(QA)

def parse_output(input_string: str, metadata: dict):
    """
    Returns list of sentences from input_string (based on line breaks)
    """
    parsed_data = input_string.split("\n")
    parsed_data = [sentence.strip() for sentence in parsed_data if len(sentence.strip()) > 10]
    return parsed_data
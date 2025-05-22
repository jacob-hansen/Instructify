import random
import re
from prompt.base_prompts import ParseOutputError

PROMPT = """You're tasked with creating a multiple-choice question about an image with the following guidelines:
- Give the correct answer as (A) and four incorrect options (B, C, D, E)
- Base your question and answers only on the given details (Fix any grammar or spelling errors that may be in the raw data)
- The asker should ask directly about the image, assuming they can see it
- The incorrect answers should be of comparable plausibility, complexity and difficulty to the correct answer

Structure your response as:
Question: [Multiple-choice question]
A. [Correct answer]
B. [Incorrect option]
C. [Incorrect option]
D. [Incorrect option]
E. [Incorrect option]"""

question_formats = [
    "Please type the <ANSWER_TYPE> of your answer",
    "Enter the <ANSWER_TYPE> corresponding to the correct answer",
    "Type the correct <ANSWER_TYPE>",
    "Please respond with the <ANSWER_TYPE> of your choice",
    "Input the <ANSWER_TYPE> that corresponds to your answer",
    "Reply with the appropriate <ANSWER_TYPE>",
    "Provide the <ANSWER_TYPE> that matches your selection",
    "Type your selected <ANSWER_TYPE>",
    "Respond with the <ANSWER_TYPE> you choose",
    "Enter your answer as a single <ANSWER_TYPE>",
    "Type the <ANSWER_TYPE> next to the correct answer",
    "Reply by typing the correct <ANSWER_TYPE>",
    "Provide the <ANSWER_TYPE> of the correct answer",
    "Input the <ANSWER_TYPE> of the answer you choose",
    "Type in the <ANSWER_TYPE> of your answer",
    "Respond with the <ANSWER_TYPE> corresponding to the right choice",
    "Type out the <ANSWER_TYPE> of your chosen answer",
    "Enter the <ANSWER_TYPE> that best fits the answer",
    "Input the appropriate <ANSWER_TYPE> for your answer",
    "Reply with the <ANSWER_TYPE> that best answers the question",
    "Please type the correct answer’s <ANSWER_TYPE>",
    "Type the <ANSWER_TYPE> that you believe is correct",
    "Provide your answer in <ANSWER_TYPE> form",
    "Input the <ANSWER_TYPE> indicating your answer",
    "Reply with the <ANSWER_TYPE> of the answer you believe is correct",
    "Type the <ANSWER_TYPE> that represents your answer",
    "Enter the <ANSWER_TYPE> from the options given",
    "Input your chosen answer’s <ANSWER_TYPE>",
    "Type the corresponding <ANSWER_TYPE> of your choice",
    "Respond by entering the <ANSWER_TYPE> of the correct option",
    "Provide the <ANSWER_TYPE> that you think is correct",
    "Type the <ANSWER_TYPE> representing the correct answer",
    "Enter the <ANSWER_TYPE> for your chosen answer",
    "Input the <ANSWER_TYPE> that you think is the answer",
    "Reply with the <ANSWER_TYPE> that corresponds to your selection",
    "Type the <ANSWER_TYPE> of what you think is correct",
    "Enter your selected answer’s <ANSWER_TYPE>",
    "Input the <ANSWER_TYPE> of the choice you believe is right",
    "Reply by typing the <ANSWER_TYPE> you select",
    "Provide the <ANSWER_TYPE> corresponding to your answer",
    "Type the <ANSWER_TYPE> associated with the correct answer",
    "Enter the <ANSWER_TYPE> representing the answer",
    "Input the <ANSWER_TYPE> of your preferred answer",
    "Reply with your chosen answer’s <ANSWER_TYPE>",
    "Type the <ANSWER_TYPE> you think is correct",
    "Enter the <ANSWER_TYPE> you choose as your answer",
    "Input the correct answer by <ANSWER_TYPE>",
    "Reply with the <ANSWER_TYPE> that fits the answer",
    "Provide the <ANSWER_TYPE> of the answer you selected",
    "Type the <ANSWER_TYPE> of the answer you believe to be correct",
    "Enter the <ANSWER_TYPE> that you think answers the question",
    "Input the <ANSWER_TYPE> that fits your answer",
    "Reply with the selected <ANSWER_TYPE>",
    "Provide your answer by typing the correct <ANSWER_TYPE>",
    "Type the <ANSWER_TYPE> that corresponds to the right answer",
    "Enter the <ANSWER_TYPE> you consider correct",
    "Input the <ANSWER_TYPE> you think is right",
    "Reply with the <ANSWER_TYPE> that you consider the answer",
    "Provide the <ANSWER_TYPE> that you have chosen",
    "Type the <ANSWER_TYPE> that fits the answer",
    "Enter the <ANSWER_TYPE> of the response you choose",
    "Input the <ANSWER_TYPE> that you select as correct",
    "Reply by providing the correct <ANSWER_TYPE>",
    "Provide the <ANSWER_TYPE> representing the answer",
    "Type the <ANSWER_TYPE> that you think is right",
    "Enter the <ANSWER_TYPE> corresponding to the answer",
    "Input the <ANSWER_TYPE> as your answer",
    "Reply with the correct answer's <ANSWER_TYPE>",
    "Provide the <ANSWER_TYPE> that represents the answer",
    "Type the <ANSWER_TYPE> of the response you believe is right",
    "Enter the <ANSWER_TYPE> as your response",
    "Input the <ANSWER_TYPE> you believe to be correct",
    "Reply with the <ANSWER_TYPE> representing the correct response",
    "Provide the <ANSWER_TYPE> of the correct response",
    "Type the <ANSWER_TYPE> that answers the question",
    "Enter the <ANSWER_TYPE> of your response",
    "Input the <ANSWER_TYPE> of the correct choice",
    "Reply with your response as a <ANSWER_TYPE>",
    "Provide the correct <ANSWER_TYPE> as your answer",
    "Type the <ANSWER_TYPE> corresponding to your choice",
    "Enter the <ANSWER_TYPE> that best represents the answer",
    "Input the <ANSWER_TYPE> of your choice",
    "Reply with the <ANSWER_TYPE> you think answers the question correctly",
    "Provide the <ANSWER_TYPE> that answers the question",
    "Type the correct <ANSWER_TYPE> of your choice",
    "Enter the <ANSWER_TYPE> that you pick",
    "Input the chosen <ANSWER_TYPE>",
    "Reply with the <ANSWER_TYPE> you believe to be correct",
    "Provide the <ANSWER_TYPE> of your answer"
]

# Multiple Choice Answer Types
answer_types = [
    ("letter", ["A", "B", "C", "D", "E"]),
    ("letter", ["A", "B", "C", "D", "E"]),
    ("letter", ["A", "B", "C", "D", "E"]),
    ("letter", ["a", "b", "c", "d", "e"]),
    ("letter", ["a", "b", "c", "d", "e"]),
    ("word", ["One", "Two", "Three", "Four", "Five"]),
    ("number", ["1", "2", "3", "4", "5"]),
    ("number", ["1", "2", "3", "4", "5"]),
    ("alphanumeric", ["A1", "B2", "C3", "D4", "E5"]),
    ("roman numeral", ["I", "II", "III", "IV", "V"]),
    ("roman numeral", ["I", "II", "III", "IV", "V"]),
    ("symbol", ["✓", "✗", "Δ", "○", "□"]),
    ("phonetic", ["Alpha", "Bravo", "Charlie", "Delta", "Echo"])
]

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
    Parse the input string for the question and answer options.

    Args:
        input_string (str): The input string containing the question and answer options.
        metadata (dict): A dictionary containing metadata (not used in this implementation).

    Returns:
        list: A list containing two elements: the formatted question and the correct answer <ANSWER_TYPE>.

    Raises:
        ParseOutputError: If the question or answer options are not found or don't match the expected format.
    """
    # Regular expression to match the question and answer options
    pattern = r"Question:\s*(.*?)\nA\.\s*(.*?)\nB\.\s*(.*?)\nC\.\s*(.*?)\nD\.\s*(.*?)\nE\.\s*(.*?)$"
    
    match = re.search(pattern, input_string, re.DOTALL)
    
    if not match:
        raise ParseOutputError("Question or answer options not found or don't match the expected format.")
    
    original_question = match.group(1).strip()
    correct_answer = match.group(2).strip()
    options = [match.group(i).strip() for i in range(2, 7)]
    
    # Randomly select how many options to use (2, 3, 4, or 5)
    num_options = random.randint(2, 5)
    
    # Select the options to use
    selected_options = options[:num_options]
    
    # Shuffle the selected options
    random.shuffle(selected_options)
    
    # Find the new position of the correct answer
    correct_answer_index = selected_options.index(correct_answer)
    
    # Format the question
    answer_type = random.choice(answer_types)
    correct_answer = answer_type[1][correct_answer_index]

    question_format = random.choice(question_formats).replace("<ANSWER_TYPE>", answer_type[0])
    formatted_question = f"{original_question} {question_format}\n"
    punctuation = random.choice([".", ")", "", "-", ":", ";"])
    for i, option in enumerate(selected_options):
        formatted_question += f"{answer_type[1][i]}{punctuation} {option}\n"
    
    return [formatted_question.strip(), correct_answer]

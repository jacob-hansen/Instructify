import random
from prompt.base_prompts import multi_turn_parsing

PROMPT = """Design a conversation of <CONVERSATION_LENGTH> turns between someone asking about an image and an assistant answering, with a slight twist. The agent asking deceptive questions in attempt to get the assistant to say something false, but the assistant responds appropriately, avoiding any false statements or deceptive content.
Importantly:
- Avoid compound questions
- Ask direct questions (e.g. "What is the capital of France?" instead of "I am in Paris, is it the capital?")
- <STRATEGY>
- The assistant NEVER makes things up, always avoids any false statements or deceptive content.

First, bullet point a few lines describing approaches for deceptive content about the image (3-5 examples). Each of the bullet points are generic ideas for twisting reality of this image in ways that are believable, describing how the situation 'could be' twisted. <EXAMPLE>After writing the list, conclude with "<FINISHED_BRAINSTORMING>".
Then write turns of conversation between the agent (asking questions) and the assistant. There should be <CONVERSATION_LENGTH> questions. Select the ideas from the best bullet point(s) you listed. Each question directly asks about some aspect of the image (or aspect which cannot be answered)
Both the questions and answers should be written in first person, as if the agent and assistant are looking at the image. Interleave the normal and deceptive questions, formatting them the same way.

Format your response as follows:
[brainstorming deceptive questions (3-5 lines)]
<FINISHED_BRAINSTORMING>
Question: [Question]
Answer: [Answer]"""

asignment_strategies = [
    "Start with a real question, and slowly transition more deceptive questions.",
    "Start with a real question, and then ask deceptive questions.",
    "Start with a deceptive question, and then ask real questions.",
    "Start with more deceptive questions, and then transition to more real questions.",
    "Ask a mix of real and deceptive questions throughout the conversation.",
    "Ask half real and half deceptive questions throughout the conversation.",
]
strategies = [
    "Ask deceptive questions about things that may not be visible in the image, requiring the assistant to recognize uncertainty.",
    "Use twisted hypotheticals that seem answerable but are impossible to respond to.",
    "Ask about non-existent details in the image.",
    "Interleave questions about real and non-existent details in the image.",
    "Exploit false assumptions or common misconceptions, like 'I see ..., is that a ...?'."
]
examples = {
    "lake": "For example, in an image of a lake, \"- No boat is mentioned, but commonly found, so asking 'What color is the boat?' or 'How many people are on a boat' is confusing\". ",
    "room": "For example, in an image of a classroom, \"- The teacher is using the chalkboard, so asking 'What is displayed on the projector screen?' is confusing\". ",
    "book": "For example, in an image with someone reading a book, \"- Asking 'What is the most exciting part of the book?' is tricky because interest or engagement is a subjective assessment and that content cannot be inferred from the image\". "
}
max_conversation_length = 5

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    conversation_length = random.randint(2, max_conversation_length)
    strategy = random.choice(strategies) + " " + random.choice(asignment_strategies)
    example = examples[random.choice(list(examples.keys()))]
    prompt_formatted = PROMPT.replace("<CONVERSATION_LENGTH>", str(conversation_length)).replace("<STRATEGY>", strategy).replace("<EXAMPLE>", example)
    return prompt_formatted, {"conversation_length": conversation_length}

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
    input_string = input_string.split("<FINISHED_BRAINSTORMING>", 1)[1]
    turns = multi_turn_parsing(input_string, metadata["conversation_length"])
    # remove questions that start with "I see" or don't end with a question mark
    filtered_turns = []
    for i in range(0, len(turns), 2):
        if "I see" == turns[i][:5]:
            continue
        if not turns[i].strip().endswith("?"):
            continue
        if "color" in turns[i]:
            continue
        filtered_turns.extend(turns[i:i+2])
    if len(filtered_turns) == 0:
        raise ParseOutputError("No questions or answers found.")
    return filtered_turns
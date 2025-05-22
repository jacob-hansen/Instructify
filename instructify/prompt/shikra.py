import re
import random
from prompt.base_prompts import multi_turn_parsing

def flip_bbox(match):
    content = match.group(1) or match.group(2)
    nums = content.split(',')
    nums = [n.strip() for n in nums]
    # Always take last 4 elements
    coords = nums[-4:]
    coords[1] = f"{(1 - float(coords[1])):.2f}"
    coords[3] = f"{(1 - float(coords[3])):.2f}"
    return f"[{', '.join(coords)}]"

coordinate_formats = {
    # Spotting with Coordinates - Questions and answers that include bounding box locations
    "spotting": """The conversation should follow a format where:
    - Questions ask about scene contents followed by requesting coordinates. The questions MUST specifically request coordinates if the coordinates are to be provided. (e.g. "The question? Please provide the coordinates of the object").
    - Answers describe objects/actions and include their precise locations as [x1, y1, x2, y2]. Coordinates must match exactly with information given.""",

    # Grounding Caption - Describing specific regions with provided coordinates
    "grounding": """The exchange should be structured as:
    - Questions asking about contents within specific regions, denoting them with their coordinates [x1, y1, x2, y2]. Coordinates must match exactly with information given.
    - Answers focusing on detailed descriptions of just those bounded regions""",
}

conversation_formats = {
    # REG (Referring Expression Generation) - Uniquely distinguishing regions
    "reg": """Format the conversation where coordinates are not used, rather every QA pair is text based as follows:
    - Questions ask for unique descriptions of specific regions to distinguish them from others
    - Answers provide distinctive features and characteristics that set the region apart""",

    # Q→A (Question Answering) - Direct question-answer without coordinates
    "q_a": """Structure the dialogue where coordinates are not used, rather every QA pair is text based as follows:
    - Ask clear, specific questions about visible contents or events in the image
    - Provide concise, factual answers based solely on what's visible""",

    # Q→CA (Question with Careful Analysis) - Detailed reasoning
    "q_ca": """The QA should be formatted where coordinates are not used, rather every QA pair is text based as follows:
    - Questions prompt for both answers and explanation of reasoning
    - Responses include step-by-step logic and detailed analysis of visual evidence""",

    # Q→CBox A (Question with Coordinates Answer) - Questions with bounding box responses
    "q_cbox_a": """Design the interaction to:
    - Frame open-ended questions about scene contents
    - Structure answers to include both natural descriptions and bounding boxes [x0,y0,x1,y1] for referenced objects"""
}

PROMPT = """You are an AI visual assistant, and you are seeing a single image. What you see are provided with captions and bounding boxes of objects in the image.

Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask diverse questions and give corresponding answers.

<CONVESATION_FORMAT>

Include questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers:
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.
Do not ask any question that cannot be answered confidently.

Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details.
Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized. You can include multiple paragraphs if necessary.

Format the question/answer pairs as follows, with no introduction or explanation:
Question: [Question]
Answer: [Answer]
Question: [Question]
Answer: [Answer]
etc."""

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    # Randomly choose whether to use coordinate-based or regular conversation format
    if random.random() < 0.5:
        select_format = random.choice(list(coordinate_formats.keys()))
        format_instructions = coordinate_formats[select_format]
        is_coordinate = True
    else:
        select_format = random.choice(list(conversation_formats.keys()))
        format_instructions = conversation_formats[select_format]
        is_coordinate = False
    
    return PROMPT.replace("<CONVESATION_FORMAT>", format_instructions), {"format": select_format}

def parse_input(input_info: list, metadata: dict):
    return "\n".join(input_info)

def parse_output(input_string: str, metadata: dict):
    input_string = re.sub(r'\[(.*?)\]|\((.*?)\)', flip_bbox, input_string)
    return multi_turn_parsing(input_string)
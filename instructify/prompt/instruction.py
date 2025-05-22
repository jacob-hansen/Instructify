import random
from prompt.base_prompts import multi_turn_parsing

PROMPT = """Design a question and answer pair as an instruction-following task about the visual details of an image. Attached are details of the image. The task should target creative or analytical responses (e.g., <EXAMPLES_INSERT>). Follow these guidlines:
- The question should be open-ended and task-specific, revealing nothing about the image.
- When considering different perspectives, remember somethings may be inverted to what the current perspective sees.
- The answer focuses on what is relevant to the task, and may only use a small fraction of the information, avoiding rambling.
- Avoid any assumptions or extrapolations beyond what you know. Everything should be grounded in the observations.
- Both the question and answer talk directly about the image, assuming both can see it.

Format the question/answer pair as follows:
Question: [Task-specific instruction, image agnostic]
Answer: [Task-specific and image-specific response]"""

tasks = [
    "writing a poem",
    "text-based puzzle",
    "creative writing",
    "describing an object in detail",
    "summarizing the scene in one sentence",
    "imagining a conversation between two characters",
    "explaining the mood of the scene",
    "creating a backstory for a character",
    "identifying key elements in the scene",
    "drawing connections between two objects in the scene",
    "writing a letter from a character's perspective",
    "describing the weather and its impact on the scene",
    "narrating a moment of action",
    "giving a motivational speech based on the scene",
    "composing a song inspired by the image",
    "creating a riddle based on a detail in the scene",
    "comparing the scene to a well-known painting",
    "predicting what happens next in the scene",
    "analyzing the relationships between characters",
    "writing a diary entry from the day of the scene",
    "describing the scene through a child's eyes",
    "describes the scene in the voice of a character or character type",
    "explaining the significance of an object",
    "telling a story in 5 sentences about the scene",
    "creating a dialogue between two unseen characters",
    "identifying hidden emotions in the scene",
    "writing a haiku inspired by the image",
    "imagining what the scene would look like at night",
    "translating the scene into a different artistic medium",
    "describing the hypothetical sounds in the scene",
    "creating a list of smells one might experience",
    "imagining a secret one of the characters holds",
    "recreating the scene with futuristic elements",
    "writing a limerick about the scene",
    "describing the lighting and its effect on mood",
    "detailing what could have happened 10 minutes before the scene",
    "writing from the perspective of an inanimate object",
    "explaining the scene to someone from another time period",
    "giving instructions for recreating the scene",
    "creating a metaphor to describe the scene",
    "describing the scene using only color-based adjectives",
    "writing a newspaper headline about the scene",
    "composing an interior monologue for a character",
    "identifying potential conflicts in the scene",
    "describing the textures and surfaces of objects",
    "creating a list of what could go wrong in the scene",
    "imagining the scene in an alternate universe",
    "describing a character's outfit in vivid detail",
    "analyzing the composition of the scene",
    "writing a speech one character might give",
    "reinterpreting the scene as a stage play",
    "writing a flashback tied to the scene",
    "explaining how the scene reflects a theme",
    "creating a dialogue based on body language",
    "describing the scene from an animal's perspective",
    "naming the scene as if it were a painting",
    "writing a letter to one of the characters",
    "imagining the scene in a completely different season",
    "writing about the scene as if it were a memory",
    "explaining the symbolism behind one object",
    "identifying hidden tensions or unresolved issues",
    "describing the emotional arc of a character",
    "writing a travel guide about the location",
    "creating an advertisement based on the scene",
    "writing a eulogy for an unseen character",
    "describing the scene in reverse order",
    "writing an apology from one character to another",
    "imagining a phone conversation about the scene",
    "explaining the impact of time on the scene",
    "creating a recipe based on something in the image",
    "writing a dialogue between two objects in the scene",
    "describing a moment of stillness in the scene",
    "writing a news report on the events in the scene",
    "imagining a celebration taking place here",
    "writing from the perspective of a lost object",
    "describing how the scene would be captured on film",
    "writing about a parallel version of the scene",
    "describing what one character might be dreaming of",
    "analyzing the facial expressions of a character",
    "writing a job interview for one of the characters",
    "explaining a tradition that might take place here",
    "imagining the same scene from a different angle",
    "creating a fictional historical event for the setting",
    "writing about the scene using only sounds and smells",
    "explaining how a child might misunderstand the scene",
    "writing a thank-you note from one character to another",
    "describing how technology would alter the scene",
    "imagining the scene in the form of a board game",
    "writing a rumor being spread about the scene",
    "describing the scene in complete darkness",
    "explaining the scene through a metaphor",
    "writing a character's final thoughts in the scene",
    "describing the moment after the scene ends",
    "explaining how the scene would change over time",
    "describing the physical sensations in the scene",
    "writing a series of text messages between characters",
    "imagining the scene as a work of graffiti",
    "explaining how the scene would feel in a dream",
    "describing a festival or event taking place here",
    "writing about the scene from the perspective of an outsider",
    "creating a python list of the objects in the image",
    "counting the number of people in the scene",
    "identifying the dominant color in the image",
    "describing the position of key objects",
    "listing all the visible types of transportation",
    "noting the time of day based on the image",
    "estimating the weather conditions from the image",
    "categorizing objects based on their size",
    "identifying whether the scene is indoors or outdoors",
    "detecting any written text or signs in the image",
    "measuring distances between key objects",
    "identifying the focus or central point of the image",
    "creating a list of potential actions happening in the image",
    "determining if the image includes any motion or movement",
    "describing the architecture visible in the image",
    "naming any visible animals in the image",
    "analyzing the clothing style of the people",
    "assessing the lighting source in the image",
    "identifying any food or drinks visible",
    "grouping objects based on their function",
    "noting any technological devices present in the image",
    "recognizing the season depicted in the image",
    "describing the furniture or interior design elements",
    "listing any visible plants or trees",
    "categorizing objects based on their material in a python dictionary",
    "identifying any signs of damage or wear in the objects",
    "suggesting improvements to the organization of objects",
    "noting any potential hazards in the image",
    "determining the cleanliness or tidiness of the scene",
    "recognizing the presence of any vehicles",
    "describing any visible shadows and their direction",
    "counting the number of distinct patterns or textures",
    "identifying the style of artwork or decorations",
    "listing any objects that appear out of place",
    "determining the focal length or depth of field",
    "analyzing the overall symmetry of the image",
    "grouping objects by color",
    "listing items that could be moved for more space",
    "assessing whether the scene appears organized or chaotic",
    "identifying any hand-held objects",
    "describing any recurring themes in the image",
    "making a list of objects that suggest a specific location",
    "identifying any visible water or liquid in the image",
    "suggesting actions to declutter the scene",
    "analyzing the arrangement of people or objects", 
    "comparing the sizes or positions of objects in the image",
    "contrasting color distribution across different parts of the image", 
    "delve deeply into the specific particulars of the image",
    "interpret the entire image",
    "write a detailed commentary on the image",
    "break down the image intricately"
]

def sample():
    """
    Samples the prompt and returns it along with the expected information for parsing.

    Returns:
        str: The prompt.
        dict: Arguments to pass to the parse function.
    """
    # sample 3 tasks
    task_sample = random.sample(tasks, 3)
    task_str = ", ".join(task_sample)
    prompt = PROMPT.replace("<EXAMPLES_INSERT>", task_str)
    return prompt, {}

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
    return multi_turn_parsing(input_string, 1)
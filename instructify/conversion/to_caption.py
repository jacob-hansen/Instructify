PROMPT = """\nCarefully think about the provided information. In an appropriate number of paragraphs to cover everything, describe observable functional, spatial or informative relationships perceivable as if one were looking at the image. Avoid speculation and describe only that which is supported by the information provided."""
data_description = """The information is a hierarchical structure representing objects in an image. With the following rules:
- Indentation shows the relationships between objects, where sequence is unimportant, but indentation and arrows indicates relation (e.g. holding or inside). 
- An X at the end of the object caption indicates that there is nothing further is associated with that object. 
- The positions of objects are provided as coordinates (where 0,0 is the bottom left, and 1,1 is the top right). 
- Pixel size represents percentage of image occupancy. 
- Object depth is an arbitrary rating where different levels show how objects are positioned "closer" (smaller values) or "further away" (larger values) relative to other objects. 
In your captions, DO NOT include any coordinates given, mask sizes, nor mention of depth value. Avoid associations of objects that are not explicitly mentioned. So, even if it is likely that it is the case, it has to be explicitly given! Carefully choose counting, either explicit values, "multiple", or "many" to avoid large large or uncertain counts. Primarily focus on relationships (described in the hierarchy) and image understanding that you can confidently infer from the data."""

def parse_input(input_info, metadata: dict, includ_box_labels=False):
    """
    Parses the input and returns the expected output.

    Args:
        input_string (Unkown): The input data to parse.
        metadata (dict): Additional information to use in parsing.

    Returns:
        Any: The parsed output.
    """
    if type(input_info) == list:
        input_info = "\n".join(input_info)
    return data_description + "\n\n" + input_info
    

def parse_output(input_string: str, metadata: dict):
    """
    Parses the output string and returns the expected output.

    Args:
        input_string (str): The output string to parse.
        metadata (dict): Additional information to use in parsing.

    Returns:
        Any: The parsed output.
    """
    result = input_string.split("\n")
    return [line.strip() for line in result if len(line.strip()) > 0]
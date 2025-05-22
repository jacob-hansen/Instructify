from prompt.base_prompts import ParseOutputError
PROMPT = """Evaluate if a question/answer pair is correct based on the given information.

Return "True" if the answer:
- Is clearly correct
- Is supported by the provided evidence
- Has a definitive, unambiguous conclusion

Return "False" if the answer:
- Is unclear or nonsensical
- Makes disputable assumptions
- Lacks supporting evidence
- Claims "cannot determine" when sufficient information exists"""

def parse_input(input_info, metadata: dict):
    return f"Input information: {input_info['input_information']}\nQuestion: {input_info['question']}\nAnswer: {input_info['answer']}"

def parse_output(input_string: str, metadata: dict):
    decision = input_string.strip().lower().split()[0]
    if decision not in ["true", "false"]:
        raise ParseOutputError("The answer must be either 'true' or 'false'.")
    return decision == "true"

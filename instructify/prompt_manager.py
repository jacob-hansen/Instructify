import os
import re
import sys
import importlib
import json
import asyncio
import random
from typing import Any, Callable, Dict, List, Tuple

class PromptManagerError:
    """
    Custom error class for PromptManager that is non-JSON serializable
    and evaluates to False in boolean contexts.
    """
    def __init__(self, message: str, error_type: str = "generic"):
        self.message = message
        self.error_type = error_type
    
    def __bool__(self) -> bool:
        return False
    
    def __str__(self) -> str:
        return f"PromptManagerError: {self.error_type}: {self.message}"
    
    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self):
        """
        Raises TypeError to prevent JSON serialization
        """
        raise TypeError(f"PromptManagerError is not JSON serializable: {self.message}")

class PromptManager:
    def __init__(self, prompt_dir: str = "prompt"):
        self.prompt_dir = prompt_dir

    def list_prompts(self) -> str:
        """
        List all available prompts in the prompt directory, including invalid ones with reasons.

        Returns:
            str: A formatted string of valid and invalid prompts.
        """
        valid_prompts = []
        invalid_prompts = {}

        for file in os.listdir(self.prompt_dir):
            if file.endswith(".py") and not file.startswith("__"):
                prompt_name = file[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"{self.prompt_dir}.{prompt_name}")
                    
                    # Check for necessary attributes/functions
                    if not hasattr(module, 'PROMPT') and not hasattr(module, 'sample'):
                        raise AttributeError("Missing PROMPT or sample function")
                    
                    valid_prompts.append(prompt_name)
                except Exception as e:
                    invalid_prompts[prompt_name] = str(e)

        # Construct the output string
        output = f"Valid Prompts: {json.dumps(valid_prompts)}\n"
        if invalid_prompts:
            output += "Invalid Prompts:\n"
            for prompt, reason in invalid_prompts.items():
                output += f"    \"{prompt}\": {reason}\n"
        
        return output

    async def run_prompt(self, prompt_path: str, input_data: Any, model_callable: Callable, max_retries: int = 3) -> Any:
        """
        Run the specified prompt with the given input data and model.
        """
        try:
            module = self._load_module(prompt_path)
        except ImportError as e:
            return PromptManagerError(
                f"Prompt '{prompt_path}' not found or failed to import. Details: {str(e)}",
                "import_error"
            )

        # Sample the prompt
        try:
            prompt, metadata = self._sample_prompt(module)
        except Exception as e:
            return PromptManagerError(
                f"Error in sampling prompt: {str(e)}",
                "sampling_error"
            )

        # Parse input
        try:
            parsed_input = self._parse_input(module, input_data, metadata)
        except Exception as e:
            return PromptManagerError(
                f"Error in parsing input: {str(e)}",
                "parsing_error"
            )

        # Run the engine and parse output
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "user", "content": prompt + "\n" + parsed_input}
                ]
                response = await model_callable(messages)
                
                try:
                    return self._parse_output(module, response, metadata)
                except Exception as e:
                    if attempt == max_retries - 1:
                        return PromptManagerError(
                            f"Error in parsing output (attempt {attempt + 1} of {max_retries}): {str(e)}\nOutput: {response}",
                            "output_parsing_error"
                        )
            except Exception as e:
                if attempt == max_retries - 1:
                    return PromptManagerError(
                        f"Error in engine execution (attempt {attempt + 1} of {max_retries}): {str(e)}",
                        "engine_error"
                    )

    async def reduce(self, information: List[str], conversation: str, model_callable: Callable, question_only: bool = True, max_retries: int = 3, verbose: bool = False) -> Tuple[List[str], List[str]]:
        """
        Run the reduce prompt with the given information and conversation, treating each sentence separately.

        Args:
            information (List[str]): List of strings representing information about the image.
            conversation (str): Conversation about the image.
            model_callable (Callable): Async callable LLM engine.
            question_only (bool): Whether to only consider questions in the conversation (recommended).
            max_retries (int): Maximum number of retries for the prompt.
            verbose (bool): Whether to print verbose output.

        Returns:
            Tuple[List[str], List[str]]: Filtered information and filtered-out information.
        """
        for i in range(max_retries):
            # Split information into sentences and keep track of original groupings
            split_info: List[Tuple[int, str]] = []
            for idx, info in enumerate(information):
                sentences = re.split(r'(?<=[.!?])\s+', info)
                split_info.extend([(idx, sentence.strip()) for sentence in sentences if sentence.strip()])

            # Prepare the information for the prompt
            sentences = [sentence for _, sentence in split_info]

            # If selecting questions only, only take 0, 2, 4, ... turns
            if question_only and type(conversation) == str:
                conversation = [turn for i, turn in enumerate(conversation.split("\n")) if i % 2 == 0]
            elif question_only:
                conversation = [turn for i, turn in enumerate(conversation) if i % 2 == 0]

            # Run the reduce prompt
            filter_inds = await self.run_prompt("reduce", {"information": sentences, "conversation": conversation}, model_callable, max_retries=max_retries)

            if not filter_inds:
                continue

            # Filter out the sentences that are in filter_inds
            remove_inds = set(filter_inds)
            filtered_sentences = [split_info[i] for i in range(len(split_info)) if i not in remove_inds]
            filtered_out_sentences = [split_info[i] for i in range(len(split_info)) if i in remove_inds]

            if verbose:  # print the ones that are filtered out
                print("Filtered out sentences:")
                for i in remove_inds:
                    print("\t- " + split_info[i][1])

            # Group filtered sentences by their original information index and rejoin
            grouped_sentences = {}
            for orig_idx, sentence in filtered_sentences:
                if orig_idx not in grouped_sentences:
                    grouped_sentences[orig_idx] = []
                grouped_sentences[orig_idx].append(sentence)

            # Group filtered-out sentences by their original information index and rejoin
            grouped_out_sentences = {}
            for orig_idx, sentence in filtered_out_sentences:
                if orig_idx not in grouped_out_sentences:
                    grouped_out_sentences[orig_idx] = []
                grouped_out_sentences[orig_idx].append(sentence)

            # Rejoin the filtered information
            filtered_information = []
            for orig_idx in sorted(grouped_sentences.keys()):
                rejoined = ' '.join(grouped_sentences[orig_idx])
                filtered_information.append(rejoined)

            # Rejoin the filtered-out information
            filtered_out_information = []
            for orig_idx in sorted(grouped_out_sentences.keys()):
                rejoined = ' '.join(grouped_out_sentences[orig_idx])
                filtered_out_information.append(rejoined)

            return filtered_information, filtered_out_information
        return [], information

    async def _check_qa_pair(
        self,
        question: str,
        answer: str,
        information: List[str],
        model_callable: Callable,
        max_retries: int
    ) -> bool:
        """
        Check if a QA pair is correct using the 'check' prompt.

        Args:
            question (str): The question part of the QA pair.
            answer (str): The answer part of the QA pair.
            information (List[str]): The full original information.
            model_callable (Callable): Async callable LLM engine.
            max_retries (int): Maximum number of retries for the prompt.

        Returns:
            bool: True if the QA pair is correct, False otherwise.
        """
        check_input = {
            "input_information": ' '.join(information),
            "question": question,
            "answer": answer
        }
        is_correct = await self.run_prompt(
            "check",
            check_input,
            model_callable,
            max_retries=max_retries
        )
        return isinstance(is_correct, bool) and is_correct

    async def _get_correct_qa_pairs(
        self,
        qa_pairs: List[Tuple[str, str]],
        information: List[str],
        model_callable: Callable,
        max_retries: int
    ) -> List[Tuple[str, str]]:
        """
        Check QA pairs and return the list of correct ones.

        Args:
            qa_pairs (List[Tuple[str, str]]): List of QA pairs.
            information (List[str]): The full original information.
            model_callable (Callable): Async callable LLM engine.
            max_retries (int): Maximum number of retries for the prompt.

        Returns:
            List[Tuple[str, str]]: List of correct QA pairs.
        """
        if not qa_pairs:
            return []

        # Check the first QA pair
        first_question, first_answer = qa_pairs[0]
        is_first_correct = await self._check_qa_pair(
            first_question,
            first_answer,
            information,
            model_callable,
            max_retries
        )

        if not is_first_correct:
            # Discard the entire conversation if the first QA pair is incorrect
            return []

        correct_qa_pairs = [(first_question, first_answer)]

        # Check the rest of the QA pairs
        for question, answer in qa_pairs[1:]:
            is_correct = await self._check_qa_pair(
                question,
                answer,
                information,
                model_callable,
                max_retries
            )
            if is_correct:
                correct_qa_pairs.append((question, answer))

        return correct_qa_pairs

    async def process(
        self,
        information: List[str],
        model_callable: Callable,
        prompt_distribution: List[Dict[str, Any]],
        reduction_threshold: float = 0.85,
        max_count: int = 15,
        min_information_length: int = 100,
        prompt_retry_rate: int = 3,
        filtering_enabled: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process the information using a distribution of prompts.

        Args:
            information (List[str]): List of strings representing information about the image.
            model_callable (Callable): Async callable LLM engine.
            prompt_distribution (List[Dict[str, Any]]): List of dictionaries containing prompt names, weights, and max samples.
            reduction_threshold (float): Maximum proportion of data that can be lost before stopping.
            max_count (int): Maximum number of iterations.
            min_information_length (int): Minimum length of the combined information string.
            prompt_retry_rate (int): Maximum number of retries for the prompt.
            filtering_enabled (bool): Whether to perform the filtering step.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing details of each processing step.
        """
        results = []
        original_info = information.copy()  # Full original information for checking
        current_info = information.copy()   # Active information for prompting and reduce

        original_info_length = sum(len(info) for info in original_info)
        consecutive_failures = 0  # Keep track of consecutive failures

        for _ in range(max_count):
            # Check if we've reached the minimum information length
            if sum(len(info) for info in current_info) < min_information_length:
                break

            # Sample a prompt based on the distribution
            prompt = self._sample_prompt_from_distribution(prompt_distribution)

            # Run the sampled prompt using current_info
            output = await self.run_prompt(
                prompt["name"],
                current_info,
                model_callable,
                max_retries=prompt_retry_rate
            )

            # Assume output is a list of strings, alternating between questions and answers
            if isinstance(output, list):
                qa_pairs = [
                    (output[i], output[i + 1]) for i in range(0, len(output) - 1, 2)
                ]
            else:
                return PromptManagerError(
                    f"Error: Expected output to be a list of strings, but got {type(output)}",
                    "output_format_error"
                )

            # If filtering is enabled, perform filtering using correct QA pairs
            if filtering_enabled:
                # Get correct QA pairs, discarding if the first is incorrect
                correct_qa_pairs = await self._get_correct_qa_pairs(
                    qa_pairs,
                    original_info,
                    model_callable,
                    prompt_retry_rate
                )
                print("Filtering", set(qa_pairs) - set(correct_qa_pairs))

                if not correct_qa_pairs:
                    consecutive_failures += 1
                    if consecutive_failures > 3:
                        break  # Stop and return what we have thus far
                    continue  # Proceed to the next iteration

                # Reset consecutive failures since we have a successful iteration
                consecutive_failures = 0

                # Prepare the conversation from correct QA pairs
                conversation = [item for pair in correct_qa_pairs for item in pair]

                # Reduce the current information using the correct QA pairs
                filtered_info, filtered_out_info = await self.reduce(
                    current_info,
                    conversation,
                    model_callable,
                    verbose=False
                )

                # Calculate the reduction ratio
                current_info_length = sum(len(info) for info in current_info)
                filtered_info_length = sum(len(info) for info in filtered_info)
                reduction_ratio = 1 - (filtered_info_length / current_info_length)

                # Store the result
                results.append({
                    "prompt_type": prompt["name"],
                    "input": current_info,
                    "output": conversation,
                    "filtered_out": filtered_out_info
                })

                # Update current_info for the next iteration
                current_info = filtered_info

                # Check if we've reached the reduction threshold
                total_reduction_ratio = 1 - (sum(len(info) for info in current_info) / original_info_length)
                if total_reduction_ratio > reduction_threshold:
                    break
            else:
                # Reset consecutive failures since we have a successful iteration
                consecutive_failures = 0
                
                # If filtering is disabled, store the correct QA pairs
                results.append({
                    "prompt_type": prompt["name"],
                    "input": current_info,
                    "output": output
                })

            # Decrement the max_samples for the used prompt if it's not None
            if "max_samples" in prompt and prompt["max_samples"] is not None:
                prompt["max_samples"] -= 1
                if prompt["max_samples"] < 1:
                    prompt_distribution = [
                        p for p in prompt_distribution if p["name"] != prompt["name"]
                    ]

            # If there are no more prompts left, break
            if not prompt_distribution:
                break

        return results

    def _sample_prompt_from_distribution(self, prompt_distribution: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sample a prompt based on the given distribution.

        Args:
            prompt_distribution (List[Dict[str, Any]]): List of dictionaries containing prompt names, weights, and max samples.

        Returns:
            Dict[str, Any]: The sampled prompt dictionary.
        """
        total_weight = sum(prompt["weight"] for prompt in prompt_distribution)
        r = random.uniform(0, total_weight)
        upto = 0
        for prompt in prompt_distribution:
            if upto + prompt["weight"] >= r:
                return prompt
            upto += prompt["weight"]
        return prompt_distribution[-1]

    def _load_module(self, prompt_path: str):
        if os.path.isabs(prompt_path):
            # Absolute file path
            if not os.path.isfile(prompt_path):
                raise ImportError(f"File not found: {prompt_path}")
            module_name = os.path.splitext(os.path.basename(prompt_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, prompt_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        else:
            # Relative path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, prompt_path.replace('.', os.path.sep) + '.py')
            if not os.path.isfile(full_path):
                full_path = os.path.join(self.prompt_dir, prompt_path.replace('.', os.path.sep) + '.py')
            if not os.path.isfile(full_path):
                raise ImportError(f"File not found: {full_path}")
            module_name = prompt_path.replace('.', '_')
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module


    def _sample_prompt(self, module):
        if hasattr(module, 'sample'):
            return module.sample()
        return getattr(module, 'PROMPT', ""), {}

    def _parse_input(self, module, input_data, metadata):
        if hasattr(module, 'parse_input'):
            return module.parse_input(input_data, metadata)
        if isinstance(input_data, str):
            return input_data
        raise PromptError("Input is not a string and no parse_input function is defined.")

    def _parse_output(self, module, output, metadata):
        if hasattr(module, 'parse_output'):
            return module.parse_output(output, metadata)
        return output

# Example Usage (and test)
async def main():
    prompt_manager = PromptManager()
    
    async def mock_model(messages, temperature):
        await asyncio.sleep(1)
        return "The capital of France is Paris. 🇫🇷"

    result = await prompt_manager.run_prompt("template", "What is the capital of France?", mock_model)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams
import random

def get_async_model(model_name, gpu_memory_utilization=0.99, engine_args={}):
    """
    Get an async model
    
    Example Engine Args:
    engine_args = {
        "max_model_len": 8000,
        "quantization": "fp8",
        "device": "cuda",
        "dtype": "float16",
        "enable_lora": True
    }
    """
    engine_args = AsyncEngineArgs(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        **engine_args
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    async def get_prompt(messages):
        messages += [{"role": "assistant", "content": ""}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = "\n".join(prompt.split("\n")[:-2]) + "\n"
        return prompt

    async def generate_response(messages, temperature=0.2, max_tokens=1000, **generation_args):
        prompt = await get_prompt(messages)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, **generation_args)
        request_id = str(random.randint(0, 10000000) + 10000000)

        # Generate response using the engine
        results_generator = engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=request_id
        )

        result = ""
        async for request_output in results_generator:
            result = request_output.outputs[0].text
        
        return result
    return generate_response
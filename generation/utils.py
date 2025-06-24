import copy
import requests
import time
import os
from loguru import logger
import openai
import anthropic
from groq import Groq
import google.genai as google_genai
import json
import random
from litellm import completion
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

load_dotenv()

DEBUG = int(os.environ.get("DEBUG", "0"))
DEBUG_VERIFIER = int(os.environ.get("DEBUG_VERIFIER", 0))
DEBUG_UNIT_TEST_GENERATOR = int(os.environ.get("DEBUG_UNIT_TEST_GENERATOR", 0))
DEFAULT_CONFIG = "configs/weaver-1-110bFuser-2-110bM.json"

KEYS = None

class AllKeysUsedError(Exception):
    """Raised when all available API keys for a specific type have been used."""
    pass


class keyHandler:
    def __init__(self, api_key_data):
        self.api_key_data = api_key_data
        self.all_api_keys = self._load_api_keys()

        self.key_indices = {key: 0 for key in self.all_api_keys}

    def _load_api_keys(self):
        if isinstance(self.api_key_data, dict):
            self.all_api_keys = self.api_key_data
        else:
            try:
                with open(self.api_key_data, "r") as file:
                    return json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"API key file '{self.api_key_data}' not found."
                )

    def get_current_key(self, api_key_type):
        if api_key_type not in self.all_api_keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Check your api_keys.json"
            )

        keys = self.all_api_keys[api_key_type]
        if not keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Check your api_keys.json"
            )

        return keys[self.key_indices[api_key_type]]

    def switch_api_keys(self, api_key_type, api_key):
        print("switching key")
        if api_key_type not in self.all_api_keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Check your api_keys.json"
            )

        keys = self.all_api_keys[api_key_type]
        if not keys:
            raise ValueError(
                f"No API keys available for '{api_key_type}'. Check your api_keys.json"
            )

        current_index = keys.index(api_key)
        print(current_index)
        # used exhausted key, most likely a behind worker
        if current_index < self.key_indices[api_key_type]:
            print("used exhausted key, most likely a behind worker")
            return keys[self.key_indices[api_key_type]]

        new_index = current_index + 1

        if new_index == len(keys):
            print(f"No more keys to switch to")
            raise AllKeysUsedError(f"used all keys")

        self.key_indices[api_key_type] = new_index

        new_key = keys[new_index]
        print(
            f"Switched key for {api_key_type} from {keys[current_index]} to {new_key}"
        )
        return new_key


def clean_messages(messages):
    messages_alt = messages.copy()
    for msg in messages_alt:
        if isinstance(msg["content"], dict) and "content" in msg["content"]:
            msg["content"] = msg["content"]["content"]
    return messages_alt


def load_config(config_path):
    """
    Load the configuration from a given file path.
    If no path is provided or the file doesn't exist, use the default configuration.
    """
    if os.path.isfile(config_path):
        with open(config_path, "r") as file:
            config_file = json.load(file)
            return config_file
    else:
        raise ValueError(
            f"config_path points to missing file. Reimport {config_path} to config directory"
        )


def format_prompt(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"{role.capitalize()}: {content}\n\n"
    prompt += "Assistant: "
    return prompt


class vllmWrapper:
    def __init__(self, model_name):
        from vllm import LLM
        from transformers import AutoTokenizer

        if DEBUG:
            logger.debug("Initializing vLLM model")
        self.model = LLM(model=model_name)

        if DEBUG:
            logger.debug("Initializing vLLM tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, model_name, messages, max_tokens, temperature, **kwargs):
        from vllm import SamplingParams

        if DEBUG:
            logger.debug(
                f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model_name}` with temperature {temperature}."
            )

        if (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        else:
            logger.info("No chat template, formatting as seen in util")
            prompt = format_prompt(messages)

        if DEBUG:
            logger.debug(f"Full prompt being sent: {prompt}")

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.model.generate([prompt], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        if DEBUG:
            logger.debug(f"Output: `{response[:50]}...`.")

        return response


def generate_together(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    output = None
    request_id = random.randint(1000, 9999)  # Generate unique request ID for tracking

    key = (
        KEYS.get_current_key("TOGETHER_API_KEY")
        if KEYS
        else os.environ.get("TOGETHER_API_KEY")
    )

    logger.info(f"[Together-{request_id}] Starting request for model: {model}")
    
    for attempt, sleep_time in enumerate([1, 2, 4, 8, 16, 32], 1):
        res = None
        try:
            if DEBUG:
                logger.debug(
                    f"[Together-{request_id}] Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}` with temperature {temperature}."
                )
                logger.debug(f"[Together-{request_id}] Full message being sent: {messages}")

            endpoint = "https://api.together.xyz/v1/chat/completions"
            time.sleep(2)

            logger.info(f"[Together-{request_id}] Attempt {attempt}: Sending request...")
            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {key}",
                },
            )
            
            if "error" in res.json():
                logger.warning(f"[Together-{request_id}] Error received:")
                logger.warning(f"[Together-{request_id}] {res.json()['error']}")

                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.error(f"[Together-{request_id}] Invalid request error - stopping attempts")
                    return None
                    
                if res.json()["error"]["type"] == "credit_limit":
                    try:
                        key = KEYS.switch_api_keys("TOGETHER_API_KEY", key)
                        logger.info(f"[Together-{request_id}] Switching API key and retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                    except AllKeysUsedError as e:
                        logger.error(f"[Together-{request_id}] Exhausted all keys for Together")
                        break

            output = res.json()["choices"][0]["message"]["content"]
            logger.info(f"[Together-{request_id}] Successfully received response")
            break
            
        except AllKeysUsedError as e:
            logger.error(f"[Together-{request_id}] Exhausted all keys for Together")
            break
        except Exception as e:
            response = "failed before response" if res is None else res
            logger.error(f"[Together-{request_id}] {e} on response: {response}")
            logger.info(f"[Together-{request_id}] Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    if output is None:
        logger.error(f"[Together-{request_id}] Failed to get response after all attempts")
        return output

    if DEBUG:
        logger.debug(f"[Together-{request_id}] Output: `{output[:50]}...`.")

    return output.strip()


def generate_openai(model, messages, max_tokens=2048, temperature=0.7, **kwargs):

    key = (
        KEYS.get_current_key("OPENAI_API_KEY")
        if KEYS
        else os.environ.get("OPENAI_API_KEY")
    )

    client = openai.OpenAI(api_key=key)

    if model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"]:
        messages = [msg for msg in messages if msg["role"] != "system"]

    for sleep_time in [1, 2, 4, 8, 16, 32, 64]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

                logger.debug(f"Full message being sent: {messages}")
            if model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"]:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    reasoning_effort="high"
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def generate_anthropic(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    key = KEYS.get_current_key("ANTHROPIC_API_KEY") if KEYS else os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Client(api_key=key)
    
    # Extract system message once outside the retry loop
    system = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
    messages_alt = [msg for msg in messages if msg["role"] != "system"]
    
    for sleep_time in [1, 2, 4, 8, 16, 32, 64]:
        try:
            if DEBUG:
                logger.debug(f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`.")

            completion = client.messages.create(
                model=model,
                system=system,
                messages=messages_alt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # If we get here, we have a successful response
            return completion.content[0].text.strip()
            
        except anthropic.RateLimitError as e:
            logger.error(e)
            if KEYS:
                try:
                    key = KEYS.switch_api_keys("ANTHROPIC_API_KEY", key)
                    client = anthropic.Client(api_key=key)
                    continue  # Try immediately with new key
                except AllKeysUsedError as e:
                    logger.error(f"Exhausted all keys for Anthropic")
                    return None
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    # If we've exhausted all retries
    return None


def generate_groq(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = Groq(
        api_key=KEYS.get_current_key("GROQ_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    return output.strip()


def generate_tgi(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    client = openai.OpenAI(base_url=model, api_key="-")  # TGI endpoint

    output = client.chat.completions.create(
        model="tgi",
        messages=clean_messages(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    return output.strip()


def convert_gemini_format(messages):
    gemini_messages = []
    system = ""

    for msg in messages:
        if system == "" and msg["role"] == "system":
            system = msg["content"]
            continue

        message_type = "user"
        if msg["role"] == "assistant":
            message_type = "model"

        current_message = {"role": message_type, "parts": [{"text": msg["content"]}]}
        gemini_messages.append(current_message)

    if system == "":
        logger.warning("No system message")

    return system, gemini_messages


def generate_google(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    import google.genai as genai
    from google.genai import types
    
    # Create client with API key
    client = genai.Client(api_key=KEYS.get_current_key("GOOGLE_API_KEY"))
    
    # Extract system message
    system = ""
    user_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            user_messages.append(msg)
    
    # Convert messages to new format
    contents = []
    for msg in user_messages:
        contents.append(types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(msg["content"])]
        ))
    
    time.sleep(5)
    output = None

    for sleep_time in [4, 8, 16]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            # Use new generate_content API
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                system_instruction=system if system else None
            )
            
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            
            output = response.text
            break

        except Exception as e:
            logger.error(e)
            if sleep_time >= 32:
                sleep_time = sleep_time + random.randint(0, 16)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if isinstance(output, str):
        output = output.strip()
    elif output == None:
        output = "No Answer"
        logger.info("Gemini could not generate an answer due to safety restrictions!")
        
    return output


def generate_bedrock(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    output = None
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            # Ensure that AWS credentials and region are set via environment variables
            os.environ["AWS_ACCESS_KEY_ID"] = KEYS.get_current_key("AWS_ACCESS_KEY_ID")
            os.environ["AWS_SECRET_ACCESS_KEY"] = KEYS.get_current_key(
                "AWS_SECRET_ACCESS_KEY"
            )
            os.environ["AWS_REGION_NAME"] = KEYS.get_current_key("AWS_REGION_NAME")

            # Call Bedrock model via litellm completion function
            response = completion(
                model=model,
                messages=clean_messages(messages),
            )

            output = response["choices"][0]["message"]["content"]
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return None

    return output.strip()

def generate_deepseek(model, messages, max_tokens=2048, temperature=0.7, **kwargs):

    client = OpenAI(
        api_key=KEYS.get_current_key("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    output = None
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            response = client.chat.completions.create(
                model=model,
                messages=clean_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            output = response.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return None

    return output.strip()

class LocalHFModel:
    def __init__(self, model, model_checkpoint=None):
        if DEBUG:
            logger.debug(f"Initializing local Hugging Face model: {model}")
            if model_checkpoint:
                logger.debug(f"Loading from checkpoint: {model_checkpoint}")
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Load the model
        if model_checkpoint:
            self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
            logger.info(f"Successfully loaded model checkpoint: {model_checkpoint}")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model)
            logger.info("No model checkpoint to load!")
        
        # Set pad_token_id if it's not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if DEBUG:
            logger.debug(f"Model loaded on device: {self.device}")

    def __call__(self, model, messages, max_tokens, temperature, **kwargs):
        if DEBUG:
            logger.debug(
                f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to local HF model `{model}` with temperature {temperature}."
            )

        # Prepare input
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            logger.info("No chat template, formatting as seen in util")
            prompt = format_prompt(messages)

        if DEBUG:
            logger.debug(f"Full prompt being sent: {prompt}")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode output
        response = self.tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        if DEBUG:
            logger.debug(f"Output: `{response[:50]}...`.")

        return response.strip()
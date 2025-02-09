import gc
import json
import re
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Callable, Dict

import torch
from omegaconf import DictConfig
from vllm import LLM, SamplingParams

from src.data import SCHEMA_MAP, BaseDataSchema
from src.logger import get_logger

logger = get_logger(__name__)


############################################ Mixin Class #################################################


class TextToDictStatus(Enum):
    SUCCESS = auto()
    INVALID_ANSWER = auto()
    REDUNDANT_KEY = auto()
    MISSING_KEY = auto()
    CONVERSION_ERROR = auto()

    def __str__(self):
        return self.name.upper()

    def score(self):
        # Mapping of status to score using a dictionary
        status_scores = {
            TextToDictStatus.SUCCESS: 5,
            TextToDictStatus.INVALID_ANSWER: 0,
            TextToDictStatus.REDUNDANT_KEY: 3,
            TextToDictStatus.MISSING_KEY: 2,
            TextToDictStatus.CONVERSION_ERROR: 1
        }
        return status_scores.get(self, 0)


class TextToDictMixin:
    def __init__(self, schema: BaseDataSchema):
        self.dtype_dict = {key: val for key, val in deepcopy(schema.dtype_dict).items()}

    def _convert_value(self, dtype, value, key):
        if dtype == int:
            return int(value)
        elif dtype == float:
            return float(value)
        elif dtype == bool:
            if value in ('true', '1', 'yes'):
                return True
            elif value in ('false', '0', 'no'):
                return False
            raise ValueError(f"Cannot convert value '{value}' for key '{key}' to bool.")
        elif dtype == str:
            return value
        elif callable(dtype):
            return dtype(value)
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {dtype}")
    
    def safe_json_loads(self, text: str) -> Dict[str, Any]:
        def fix_json(input_string: str) -> str:
            # Add double quotes around keys
            fixed_string = re.sub(r'(?<!")([a-zA-Z_][a-zA-Z0-9_ -]*)(?=\s*:)', r'"\1"', input_string)
            # Add double quotes around unquoted string values
            fixed_string = re.sub(r'(?<=:\s)([a-zA-Z_][a-zA-Z0-9_-]*)(?![",\]}])', r'"\1"', fixed_string)
            return fixed_string
        string = fix_json(text)
        result = json.loads(string)
        return result
    
    def manual_json_loads(self, text: str) -> Dict[str, Any]:
        """
        Manually parse a JSON-like string into a dictionary, handling commas within values.

        Args:
            text (str): The input string to parse.

        Returns:
            Dict[str, Any]: The parsed key-value pairs.
        """
        manual_text = re.sub(r'[\\/]', '', text.strip(" \n{}"))
        # # Regex pattern to match key-value pairs, handling quoted values with commas
        # pattern = r'"(?P<key>[\w\s\-\(\)]+)"\s*:\s*(?P<value>"[^"]*"|\d+|true|false|null|[\w\-]+),?'
        pattern = r'"(?P<key>[^"]+)"\s*:\s*(?P<value>"[^"]*"|\d+|true|false|null|[^",}\s]+),?'
        compiled_pattern = re.compile(pattern)
        
        result = {}
        for match in compiled_pattern.finditer(manual_text):
            key = match.group("key").strip()
            value = match.group("value").strip().strip('"')
            
            # Preserve the original case for string values, unless conversion is needed
            result[key] = value
        
        return result

    def search_key(self, target_key, keys):
        """
        Search for the target key in the list of keys, ignoring case, spaces, slashes, and underscores.

        Args:
            target_key (str): The key to search for.
            keys (iterable): A collection of keys to search within.

        Returns:
            bool: True if a match is found, otherwise False.
        """
        # Normalize the target key
        normalized_target = target_key.lower().replace(" ", "").replace("/", "").replace("_", "")

        for key in keys:
            # Normalize each key in the keys list
            normalized_key = key.lower().replace(" ", "").replace("/", "").replace("_", "")
            if normalized_target == normalized_key:
                return key
        return None
    
    def get_valid_dict(self, pred_dict, dtype_dict) -> Dict[str, Any]:
        # convert dtype 
        status = TextToDictStatus.SUCCESS
        for key, dtype in dtype_dict.items():
            gen_key = self.search_key(key, pred_dict.keys())
            if not gen_key:
                continue
            try:
                pred_dict[gen_key] = self._convert_value(dtype, pred_dict[gen_key], gen_key)
                pred_dict[key] = pred_dict.pop(gen_key)
            except ValueError:
                status = TextToDictStatus.CONVERSION_ERROR
                pred_dict[key] = None
            
        # checking missing key
        missing_keys = set(dtype_dict.keys()) - set(pred_dict.keys())
        if missing_keys:
            # print(f"Missing keys: {missing_keys}")
            for key in missing_keys:
                pred_dict[key] = None
            status = TextToDictStatus.MISSING_KEY
        
        # checking redundant key
        redundant_keys = set(pred_dict.keys()) - set(dtype_dict.keys())
        if redundant_keys:
            if status == TextToDictStatus.SUCCESS:
                status = TextToDictStatus.REDUNDANT_KEY
        
        result = {key: pred_dict[key] for key in dtype_dict}
        return {"status": status, "result": result}
    
    def post_process(self, text: str) -> Dict[str, Any]:

        quote_pattern = r'\{[^{}]*\}'
        matches = re.findall(quote_pattern, text, re.DOTALL)
        if not matches:
            return {"status": TextToDictStatus.INVALID_ANSWER, "result": {}}

        # only use the first json object
        content = matches[0]
        
        try:
            result_dict = self.safe_json_loads(content)
        except json.JSONDecodeError:
            result_dict = self.manual_json_loads(content)
            
        valid_result = self.get_valid_dict(result_dict, self.dtype_dict)
        
        return valid_result

############################################ Client Class #################################################


class Client(ABC):

    @abstractmethod
    def generate(self, prompts):
        pass


class VllmClient(Client):
    def __init__(self, config: DictConfig):
        # Initialize LLM using vllm_params from the config
        vllm_params = config.vllm_params
        self.llm = LLM(**vllm_params)

        # Initialize sampling parameters from sampling_params in the config
        sampling_config = config.sampling_params
        self.sampling_params = SamplingParams(**sampling_config)

    def generate(self, prompts, sampling_params=None):
        if not sampling_params:
            sampling_params = self.sampling_params
        # Generate outputs using the configured sampling parameters
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)
        output_texts = [output.outputs[0].text for output in outputs]
        return output_texts

    def close(self):
        """Clean up any resources, such as deleting the LLM instance and cleaning up GPU memory."""
        logger.info("Cleaning up VllmClient resources...")

        # Manually delete the LLM instance and clear references
        del self.llm
        gc.collect()  # Run garbage collection to clean up any lingering references

        # Manually clear GPU memory if using PyTorch (or another framework with CUDA)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cleared.")
        else:
            logger.info("No GPU found, skipping CUDA memory cleanup.")

        logger.info("VllmClient resources cleaned up.")

    # Context management methods
    def __enter__(self):
        """Enter the runtime context related to this object."""
        logger.info("Entering context for VllmClient")
        # Return the client instance for use within 'with' block
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        logger.info("Exiting context for VllmClient")
        # Clean up resources when the 'with' block is exited
        self.close()
        # Handle any exceptions raised in the 'with' block if needed
        if exc_type:
            logger.error(f"Exception occurred: {exc_type.__name__} - {exc_value}")
        return False  # Propagate the exception if any


class SynClient(VllmClient, TextToDictMixin):

    def __init__(self, config: DictConfig, schema):
        VllmClient.__init__(self, config)
        TextToDictMixin.__init__(self, schema)

    def generate(self, prompts, sampling_params=None):
        if not sampling_params:
            sampling_params = self.sampling_params
        texts = super().generate(prompts, sampling_params)
        # logger.debug(f"Generated texts: {texts}")
        dicts = [self.post_process(text) for text in texts]
        # logger.debug(f"Generated dicts: {dicts}")

        results = []
        for p, t, d in zip(prompts, texts, dicts):
            r = {"prompt": p, "answer": t}
            r.update(d)
            results.append(r)
        return results


if __name__ == "__main__":

    def test_text2dict():
        """
        Test the TextToDictMixin's post_process method with a sample text.
        """
        schema = SCHEMA_MAP["adult"]
        mixin = TextToDictMixin(schema.dtype_dict)

        text = (
            "Here's a new sample text that attempts to approximate the key patterns observed in the provided samples:\n\n"
            "'''  the state of AGE is 32.00, the state of CAPITAL-GAIN is 0.00, "
            "the state of CAPITAL-LOSS is 0.00, the state of CLASS is 0, "
            "the state of EDUCATION is hs-grad, the state of EDUCATION-NUM is 9.00, "
            "the state of FNLWGT is 228265.00, the state of HOURS-PER-WEEK is 30.00, "
            "the state of MARITAL-STATUS is never-married, the state of NATIVE-COUNTRY is united-states, "
            "the state of OCCUPATION is handlers-cleaners, the state of RACE is white, "
            "the state of RELATIONSHIP is own-child, the state of SEX is female, "
            "the state of WORKCLASS is private  ''' \n\n\n\n\n\n\n "
        )

        expected_output = {
            "AGE": 32.00,
            "CAPITAL-GAIN": 0.00,
            "CAPITAL-LOSS": 0.00,
            "CLASS": 0,
            "EDUCATION": "hs-grad",
            "EDUCATION-NUM": 9.00,
            "FNLWGT": 228265.00,
            "HOURS-PER-WEEK": 30.00,
            "MARITAL-STATUS": "never-married",
            "NATIVE-COUNTRY": "united-states",
            "OCCUPATION": "handlers-cleaners",
            "RACE": "white",
            "RELATIONSHIP": "own-child",
            "SEX": "female",
            "WORKCLASS": "private"
        }

        result = mixin.post_process(text)

        print(result)
        print("-"*50)
        print(expected_output)
        print("-"*50)
        print(set(result.keys()) - set(expected_output.keys()))
        print("-"*50)
        print(set(expected_output.keys()) - set(result.keys()))
        print("-"*50)
        assert set(result.keys()) == set(expected_output.keys(
        )), "Test failed! Keys in the result do not match the expected keys."

        assert all(result[key] == expected_output[key]
                   for key in expected_output), "Test failed! Values in the result do not match the expected values."

    test_text2dict()

# SPDX-License-Identifier: Apache-2.0
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena

TODO: Implement CustomDataset to parse a JSON file and convert its contents into
SampleRequest instances, similar to the approach used in ShareGPT.
"""

import base64
import io
import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional, Union

from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MIN_INPUT_TOKENS = 4
MAX_INPUT_TOKENS = 16200
MAX_OUTPUT_TOKENS = -1

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Union[MultiModalDataDict, dict]] = None
    lora_request: Optional[LoRARequest] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None

    def apply_multimodal_chat_transformation(
        self, prompt: str, mm_content: Optional[MultiModalDataDict] = None
    ) -> list[dict]:
        """
        Transform a prompt and optional multimodal content into a chat format.
        This method is used for chat models that expect a specific conversation
        format.
        """
        content = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            content.append(mm_content)
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError("load_data must be implemented in subclasses.")

    def get_random_lora_request(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Optionally select a random LoRA request and return its associated
        tokenizer.

        This method is used when LoRA parameters are provided.  It randomly
        selects a LoRA based on max_loras and retrieves a cached tokenizer for
        that LoRA if available. Otherwise, it returns the base tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerBase): The base tokenizer to use if no
            LoRA is selected.  max_loras (Optional[int]): The maximum number of
            LoRAs available. If None, LoRA is not used.  lora_path
            (Optional[str]): Path to the LoRA parameters on disk. If None, LoRA
            is not used.

        Returns:
            tuple[Optional[LoRARequest], AnyTokenizer]: A tuple where the first
            element is a LoRARequest (or None if not applicable) and the second
            element is the tokenizer associated with the LoRA request (or the
            base tokenizer).
        """
        if max_loras is None or lora_path is None:
            return None, tokenizer

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, max_loras)
        lora_request = LoRARequest(
            lora_name=str(lora_id),
            lora_int_id=lora_id,
            lora_path=lora_path_on_disk(lora_path),
        )
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        # Return lora_request and the cached tokenizer if available; otherwise,
        # return the base tokenizer
        return lora_request, lora_tokenizer_cache[lora_id] or tokenizer

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase, num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(self, requests: list[SampleRequest], num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests, k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int,
    max_prompt_len: int,
    max_output_len: int,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    output_too_long = (max_output_len > 0) and (output_len > max_output_len)

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long or output_too_long)


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: dict[int, AnyTokenizer] = {}


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    For a PIL.Image.Image input:
      - Converts the image to RGB.
      - Saves the image as a JPEG in-memory.
      - Encodes the JPEG data as a base64 string.
      - Returns a dictionary with the image as a base64 data URL.

    For a string input:
      - Treats the string as a URL or file path.
      - Prepends "file://" if the string doesn't start with "http://" or
        "file://".
      - Returns a dictionary with the image URL.

    Raises:
      ValueError: If the input is neither a PIL.Image.Image nor a string.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

    if isinstance(image, str):
        image_url = image if image.startswith(("http://", "file://")) else f"file://{image}"
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(f"Invalid image input {image}. Must be a PIL.Image.Image or str.")


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, min_tokens: int, max_tokens: int, max_output: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_output = max_output
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [entry for entry in self.data if "conversations" in entry and len(entry["conversations"]) >= 2]
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )

            lora_request, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path
            )
            prompt_len = len(tokenizer(prompt).input_ids)
            new_output_len = len(tokenizer(completion).input_ids) if output_len is None else output_len
            if not is_valid_sequence(
                prompt_len,
                new_output_len,
                min_len=self.min_tokens,
                max_prompt_len=self.max_tokens,
                max_output_len=self.max_output,
                skip_min_output_len_check=output_len is not None,
            ):
                continue
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(prompt, None)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    lora_request=lora_request,
                )
            )
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# HuggingFace/CROZ Dataset Implementation
# -----------------------------------------------------------------------------


class HuggingFaceDataset(BenchmarkDataset):
    """
    Dataset class for processing a HuggingFace dataset with conversation data
    and optional images.
    """

    def __init__(
        self,
        min_tokens: int,
        max_tokens: int,
        max_output: int,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_output = max_output

        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided for loading data.")

        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        if self.data.features is None or "conversations" not in self.data.features:
            raise ValueError(
                "HuggingFaceDataset currently only supports datasets with "
                "a 'conversations' column like lmms-lab/LLaVA-OneVision-Data. "
                "Please consider contributing if you would like to add "
                "support for additional dataset formats."
            )
        # Shuffle and filter examples with at least 2 conversations.
        self.data = self.data.shuffle(seed=self.random_seed).filter(lambda x: len(x["conversations"]) >= 2)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        is_croz_dataset: bool,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break

            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            if is_croz_dataset:
                prompt_len = int(conv[0]["token_count"])
                output_len = int(conv[1]["token_count"])
            else:
                prompt_len = len(tokenizer(prompt).input_ids)
                output_len = len(tokenizer(completion).input_ids)

            assert isinstance(output_len, int) and output_len > 0

            if not is_valid_sequence(
                prompt_len, output_len, self.min_tokens, self.max_tokens, self.max_output
            ):
                continue

            mm_content = process_image(item["image"]) if "image" in item else None
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len and output len
                prompt = self.apply_multimodal_chat_transformation(prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests

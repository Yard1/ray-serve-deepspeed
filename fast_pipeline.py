# Based on https://huggingface.co/databricks/dolly-v2-12b/blob/main/instruct_pipeline.py

import logging
import re
from collections import UserDict
from typing import List

import torch
from transformers import PreTrainedTokenizer
from transformers.utils import ModelOutput

logger = logging.getLogger(__name__)

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(
            f"Expected only a single token for '{key}' but found {token_ids}"
        )
    return token_ids[0]


class FastPipeline:
    """Essentially a Transformers Pipeline, stripped down to bare essentials +
    InstructPipeline logic."""

    def __init__(self, model, tokenizer, device=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        if device is not None and not (isinstance(device, int) and device < 0):
            self.model.to(device)

        if device is None:
            # `accelerate` device map
            hf_device_map = getattr(self.model, "hf_device_map", None)
            if hf_device_map is not None:
                # Take the first device used by `accelerate`.
                device = next(iter(hf_device_map.values()))
            else:
                device = -1

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

    def _sanitize_parameters(self, return_full_text: bool = None, **generate_kwargs):
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (
                token
                for token in self.tokenizer.additional_special_tokens
                if token.startswith(RESPONSE_KEY)
            ),
            None,
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(
                    self.tokenizer, tokenizer_response_key
                )
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id,
        }

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text

        return preprocess_params, forward_params, postprocess_params

    def __call__(self, inputs, **kwargs):
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        model_outputs = self.forward(model_inputs, **forward_params)
        model_outputs = self._ensure_tensor_on_device(
            model_outputs, device=torch.device("cpu")
        )

        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device)
                for name, tensor in inputs.items()
            }
        elif isinstance(inputs, UserDict):
            return UserDict(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple(
                [self._ensure_tensor_on_device(item, device) for item in inputs]
            )
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype in {
                torch.float16,
                torch.bfloat16,
            }:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def preprocess(self, instruction_text, **generate_kwargs):
        if isinstance(instruction_text, str):
            prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(
                instruction=instruction_text
            )
        else:
            prompt_text = [
                PROMPT_FOR_GENERATION_FORMAT.format(instruction=text)
                for text in instruction_text
            ]
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs

    def forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        out_b = generated_sequence.shape[0]
        generated_sequence = generated_sequence.reshape(
            in_b, out_b // in_b, *generated_sequence.shape[1:]
        )

        instruction_text = model_inputs.pop("instruction_text")
        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "instruction_text": instruction_text,
        }

    def postprocess(
        self,
        model_outputs,
        response_key_token_id,
        end_key_token_id,
        return_full_text: bool = False,
    ):
        generated_sequences = model_outputs["generated_sequence"]
        instruction_texts = model_outputs["instruction_text"]

        records = []

        for generated_sequence in generated_sequences:
            instruction_text = instruction_texts[0]
            generated_sequence: List[List[int]] = generated_sequence.numpy().tolist()
            for sequence in generated_sequence:
                # The response will be set to this variable if we can identify it.
                decoded = None

                # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
                if response_key_token_id and end_key_token_id:
                    # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
                    # prompt, we should definitely find it.  We will return the tokens found after this token.
                    try:
                        response_pos = sequence.index(response_key_token_id)
                    except ValueError:
                        logger.warn(
                            f"Could not find response key {response_key_token_id} in: {sequence}"
                        )
                        response_pos = None

                    if response_pos:
                        # Next find where "### End" is located.  The model has been trained to end its responses with this
                        # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                        # this token, as the response could be truncated.  If we don't find it then just return everything
                        # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                        try:
                            end_pos = sequence.index(end_key_token_id)
                        except ValueError:
                            end_pos = None

                        decoded = self.tokenizer.decode(
                            sequence[response_pos + 1 : end_pos]
                        ).strip()

                if not decoded:
                    # Otherwise we'll decode everything and use a regex to find the response and end.

                    fully_decoded = self.tokenizer.decode(sequence)

                    # The response appears after "### Response:".  The model has been trained to append "### End" at the
                    # end.
                    m = re.search(
                        r"#+\s*Response:\s*(.+?)#+\s*End",
                        fully_decoded,
                        flags=re.DOTALL,
                    )

                    if m:
                        decoded = m.group(1).strip()
                    else:
                        # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                        # return everything after "### Response:".
                        m = re.search(
                            r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL
                        )
                        if m:
                            decoded = m.group(1).strip()
                        else:
                            logger.warn(f"Failed to find response in:\n{fully_decoded}")

                # If the full text is requested, then append the decoded text to the original instruction.
                # This technically isn't the full text, as we format the instruction in the prompt the model has been
                # trained on, but to the client it will appear to be the full text.
                if return_full_text:
                    decoded = f"{instruction_text}\n{decoded}"

                rec = {"generated_text": decoded}

                records.append(rec)

        return records

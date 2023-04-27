# Based on https://huggingface.co/databricks/dolly-v2-12b/blob/main/instruct_pipeline.py

import logging
import re
from typing import List

from ._base import BasePipeline
from .utils import get_special_token_id

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


class DollyV2Pipeline(BasePipeline):
    """Essentially a Transformers Pipeline, stripped down to bare essentials +
    InstructPipeline logic."""

    def __init__(
        self, model, tokenizer, prompt_format=None, device=None, stopping_tokens=None
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            prompt_format
            if prompt_format is not None
            else PROMPT_FOR_GENERATION_FORMAT,
            device,
            stopping_tokens,
        )

    def preprocess(self, instruction_text, **generate_kwargs):
        if isinstance(instruction_text, str):
            prompt_text = self.prompt_format.format(instruction=instruction_text)
        else:
            prompt_text = [
                self.prompt_format.format(instruction=text) for text in instruction_text
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

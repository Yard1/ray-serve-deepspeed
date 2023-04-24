import torch
from ._base import BasePipeline

SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


PROMPT_FOR_GENERATION_FORMAT = SYSTEM_PROMPT + "<|USER|>{instruction}<|ASSISTANT|>"

class StableLMPipeline(BasePipeline):
    def __init__(self, model, tokenizer, device=None) -> None:
        super().__init__(model, tokenizer, device)
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                stop_ids = [50278, 50279, 50277, 1, 0]
                for stop_id in stop_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    def preprocess(self, instruction_text, **generate_kwargs):
        if isinstance(instruction_text, str):
            instruction_text = [instruction_text]
        prompt_text = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=text)
            for text in instruction_text
        ]
        if not self.tokenizer.pad_token or self.tokenizer.pad_token_id < 0:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        return {"inputs": inputs, "instruction_text": instruction_text}

    def forward(self, model_inputs, **generate_kwargs):
        inputs = model_inputs["inputs"]
        instruction_text = model_inputs["instruction_text"]
        for t in inputs:
            if torch.is_tensor(inputs[t]):
                inputs[t] = inputs[t].to(self.model.device)

        generate_kwargs = {**inputs, **generate_kwargs}
        generated_sequence = self.model.generate(
            **generate_kwargs,
            stopping_criteria=self.stopping_criteria
        )
        return {
            "generated_sequence": generated_sequence,
            "instruction_text": instruction_text,
        }

    def postprocess(self, model_outputs, **generate_kwargs):
        tokens = model_outputs["generated_sequence"]
        instruction_text = model_outputs["instruction_text"]
        decoded = []
        for token_unwrapped in tokens:
            decoded.append(self.tokenizer.decode(token_unwrapped, skip_special_tokens=True))
        return [response[response.find(instruction_text) + len(instruction_text):] for response, instruction_text in zip(decoded, instruction_text)]

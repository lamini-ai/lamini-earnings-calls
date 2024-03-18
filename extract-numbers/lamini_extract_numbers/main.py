from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.base_prompt_object import PromptObject

from typing import Union, Iterator, AsyncIterator
from tqdm import tqdm

import asyncio
import jsonlines

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    earnings_calls = load_earnings_calls()

    calls_with_numbers = ExtractNumbersPipeline().call(earnings_calls)

    await save_earnings_calls(calls_with_numbers)


async def load_earnings_calls():
    path = "/app/lamini-extract-numbers/data/earnings-transcripts.jsonl"

    with jsonlines.open(path) as reader:
        for line in reader:
            logger.info(f"Loaded earnings call for {line['ticker']}")
            yield PromptObject(prompt="", data=line)


class ExtractNumbersPipeline(GenerationPipeline):
    def __init__(self):
        super().__init__()

        self.number_extractor = NumberExtractor()

    def forward(self, x):
        x = self.number_extractor(x)
        return x


class NumberExtractor(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.1", max_tokens=20
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompts = self.transform_prompt(prompt)
        results = super().generate(
            prompts,
            output_type={"has_value": "bool", "value": "float", "units": "str"},
            *args,
            **kwargs,
        )
        processed_results = self.process_results(results)
        return processed_results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated numbers for {result}")
            if result is None:
                continue

            if "value" not in result.response:
                continue

            if "units" not in result.response:
                continue

            yield result

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            logger.info(f"Transforming prompt for {prompt.data['answer']}")
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        prompt = "<s>[INSTR] You are a financial analyst with extensive experience at Goldman Sachs."
        prompt += "You are reading the earnings call transcript for the following company:\n\n"
        prompt += "====================\n\n"
        prompt += get_company_info(chunk)
        prompt += "====================\n\n"
        prompt += "Consider the numbers in the transcript. "
        prompt += "Output a JSON object with the following fields:\n\n"
        prompt += " {'has_value' : bool, 'value': float, 'units': str}\n\n"
        prompt += "For percentage values, convert to a decimal value - e.g. 50% is value: 50.0, units: 'percent', not value: 0.5, units: 'percent'\n\n"
        prompt += "For dollar values - e.g. 50 million is value: 50, units: 'million', not value: 50000000, units: 'dollar'\n\n"
        prompt += "Extract the value (up to 2 decimal points) and the units from the following question and answer:\n\n"
        prompt += "====================\n\n"
        prompt += chunk.data["question"] + "\n"
        prompt += chunk.data["answer"] + "\n"
        prompt += "====================\n\n"
        prompt += "[/INSTR]"

        return prompt


def get_company_info(chunk):
    # info = f"Company: {chunk.data['exchange']}\n"
    info = f"Ticker: {chunk.data['ticker']}\n"
    info += f"Date: {chunk.data['date']}\n"
    info += f"Quarter: {chunk.data['q']}\n"
    return info


async def save_earnings_calls(answers):
    path = "/app/lamini-extract-numbers/data/earnings-calls-with-numbers.jsonl"

    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving calls", unit=" calls")
        async for answer in answers:
            answer = {
                "ticker": answer.data["ticker"],
                "date": answer.data["date"],
                "q": answer.data["q"],
                "question": answer.data["question"],
                "answer": answer.data["answer"],
                "transcript": answer.data["transcript"],
                "has_value": answer.response["has_value"],
                "value": answer.response["value"],
                "units": answer.response["units"],
            }
            writer.write(answer)
            pbar.update()


asyncio.run(main())

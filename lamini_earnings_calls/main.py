from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.base_prompt_object import PromptObject

import jsonlines

import collections
import asyncio
from tqdm import tqdm

from typing import Union, Iterator, AsyncIterator

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    earnings_calls = load_earnings_calls()

    answers = QuestionAnswerPipeline().call(earnings_calls)

    await save_answers(answers)


async def load_earnings_calls():
    path = "/app/lamini-earnings-calls/data/earnings-transcripts.jsonl"

    with jsonlines.open(path) as reader:
        for line in reader:
            logger.info(f"Loaded earnings call for {line['ticker']}")
            yield PromptObject(prompt="", data=line)


class QuestionAnswerPipeline(GenerationPipeline):
    def __init__(self):
        super(QuestionAnswerPipeline, self).__init__()

        self.question_generator = QuestionGenerator()
        self.answer_generator = AnswerGenerator()

    def forward(self, x):
        x = self.question_generator(x)
        x = self.answer_generator(x)
        return x


class QuestionGenerator(GenerationNode):
    def __init__(self):
        super(QuestionGenerator, self).__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.1", max_tokens=150
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompts = self.transform_prompt(prompt)
        results = super(QuestionGenerator, self).generate(
            prompts,
            output_type={
                "question_1": "string",
                "question_2": "string",
                "question_3": "string",
            },
            *args,
            **kwargs,
        )
        processed_results = self.process_results(results)
        return processed_results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated question for {result}")
            if result is None:
                continue
            questions = result.response["question_1"], result.response["question_2"], result.response["question_3"]
            for question in questions:
                result = PromptObject(prompt=question, data=result.data.copy())
                yield result

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            chunks = chunk_prompt(prompt)
            for chunk in chunks:
                chunk.prompt = self.make_prompt(chunk)
                logger.info(f"Generating question for {chunk.data['ticker']}, {chunk.data['q']}")
                yield chunk

    def make_prompt(self, chunk):
        prompt = (
            "<s>[INSTR]You are a financial analyst with extensive experience at Goldman Sachs."
        )
        prompt += "You are reading the earnings call transcript for the following company:\n\n"
        prompt += "====================\n\n"
        prompt += get_company_info(chunk) + "\n"
        prompt += "====================\n\n"
        prompt += (
            "You are reading the following section of the earnings call transcript:\n\n"
        )
        prompt += "====================\n\n"
        prompt += chunk.data["transcript"]
        prompt += "====================\n\n"
        prompt += "Consider the numbers in the transscript. "
        prompt += "Ask three questions about the numbers in the transcript that require precise answers. "
        prompt += "Only ask questions that can be answered using the transcript."
        prompt +="[/INSTR]"

        return prompt


def chunk_prompt(prompt):
    transcript = prompt.data["transcript"]
    chunk_size = 4096
    chunk_step = 2048

    for i in range(0, len(transcript), chunk_step):
        chunk = transcript[i : i + chunk_size]
        chunked_data = prompt.data.copy()
        chunked_data["transcript"] = chunk
        prompt_object = PromptObject(prompt=prompt.prompt, data=chunked_data)

        yield prompt_object


def get_company_info(chunk):
    info = f"Company: {chunk.data['exchange']}\n"
    info += f"Ticker: {chunk.data['ticker']}\n"
    info += f"Date: {chunk.data['date']}\n"
    info += f"Quarter: {chunk.data['q']}\n"
    return info


class AnswerGenerator(GenerationNode):
    def __init__(self):
        super(AnswerGenerator, self).__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.1", max_tokens=150
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompts = self.transform_prompt(prompt)
        results = super(AnswerGenerator, self).generate(prompts, *args, **kwargs)
        processed_results = self.process_results(results)
        return processed_results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated answer for {result}")
            if result is None:
                continue
            yield result

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            prompt.data["question"] = prompt.prompt
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        prompt = (
            "<s>[INSTR] You are a financial analyst with extensive experience at Goldman Sachs."
        )
        prompt += "You are reading the earnings call transcript for the following company:\n\n"
        prompt += "====================\n\n"
        prompt += get_company_info(chunk)
        prompt += "====================\n\n"
        prompt += (
            "You are reading the following section of the earnings call transcript:\n\n"
        )
        prompt += "====================\n\n"
        prompt += chunk.data["transcript"] + "\n"
        prompt += "====================\n\n"
        prompt += "Consider the numbers in the transscript. "
        prompt += "If the answer to the question cannot be found in the transcript, reply that you do not know. "
        prompt += "Answer the following questions about the numbers in the transcript. "
        prompt += chunk.prompt
        prompt += "[/INSTR]"

        return prompt


async def save_answers(answers):
    path = "/app/lamini-earnings-calls/data/answers.jsonl"

    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving answers", unit=" answers")
        async for answer in answers:
            answer = {
                "ticker": answer.data["ticker"],
                "q": answer.data["q"],
                "date": answer.data["date"],
                "transcript": answer.data["transcript"],
                "prompt": answer.prompt,
                "question": answer.data["question"],
                "answer": answer.response["output"],
            }
            writer.write(answer)
            pbar.update()


asyncio.run(main())

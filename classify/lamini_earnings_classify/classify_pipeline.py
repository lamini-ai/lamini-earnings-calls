import asyncio
import logging
from typing import AsyncIterator, Iterator, List, Union

import jsonlines
import lamini
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini_classifier import LaminiClassifier
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    lamini.batch_size = 128
    classifier = LaminiClassifier.load("models/classifier.lamini")

    examples = load_examples()

    predictions = ClassificationPipeline(classifier).call(examples)

    await save_predictions(predictions)


def load_examples():
    path = "/app/lamini-earnings-classify/data/answers.jsonl"
    with jsonlines.open(path) as reader:
        for example in reader:
            yield PromptObject("", data=example)


class ClassificationPipeline(GenerationPipeline):
    def __init__(self, classifier: LaminiClassifier):
        super(ClassificationPipeline, self).__init__()
        self.embedding_generator = EmbeddingGenerator(classifier)

    def forward(self, x):
        x = self.embedding_generator(x)
        return x


class EmbeddingGenerator(EmbeddingNode):
    def __init__(self, classifier: LaminiClassifier):
        super(EmbeddingGenerator, self).__init__()
        self.classifier = classifier

    async def transform_prompt(
        self, prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]
    ):
        # print("prompt1", prompt)
        if isinstance(prompt, Iterator):
            for a in prompt:
                # print(f"self.preprocess {self.preprocess}")
                # print(f"self.preprocess {self.preprocess is None}")
                if self.preprocess:
                    # print("preprocess OOGABOOGA")
                    mod_a = self.preprocess(a)
                    if mod_a is not None:
                        a = mod_a
                yield a
        elif isinstance(prompt, AsyncIterator):
            async for a in prompt:
                # print(f"self.preprocess2 {self.preprocess}")

                if self.preprocess:
                    # print("preprocess OOGABOOGA2")
                    mod_a = self.preprocess(a)
                    if mod_a is not None:
                        a = mod_a
                yield a
        else:
            raise Exception("Invalid prompt type")

    def preprocess(self, prompt: PromptObject):
        # print("preprocess OOGABOOGA3")

        prompt.prompt = self.form_prompt(prompt.data)
        # print("prompt2", prompt.prompt)

    async def batch(self, examples):
        batch = []

        async for example in examples:
            batch.append(example)

            if len(batch) == lamini.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    async def process_results(self, results: AsyncIterator[PromptObject]):
        batches = self.batch(results)
        async for batch in batches:
            result = self.postprocess(batch)
            for r in result:
                yield r

    def postprocess(self, prompt_list: List[PromptObject]):
        probabilities = self.classifier.classify_from_embedding(
            [prompt.response for prompt in prompt_list]
        )
        for prompt, probability in zip(prompt_list, probabilities):
            prompt.data["predictions"] = probability
        return prompt_list

    def form_prompt(self, example):
        prompt = "Consider this earnings call transcript:\n"
        prompt += "====================\n"
        prompt += example["transcript"] + "\n"
        prompt += "====================\n"
        prompt += f"Date of the call: {example['date']}\n"
        prompt += f"Ticker: {example['ticker']}\n"
        prompt += f"Quarter: {example['q']}\n"
        prompt += "====================\n"
        prompt += example["question"] + "\n"
        prompt += "====================\n"
        prompt += example["answer"] + "\n"
        return prompt


async def save_predictions(predictions):
    path = "/app/lamini-earnings-classify/data/predictions.jsonl"

    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving predictions", unit=" predictions")
        async for prediction in predictions:
            writer.write(prediction.data)
            pbar.update()


asyncio.run(main())

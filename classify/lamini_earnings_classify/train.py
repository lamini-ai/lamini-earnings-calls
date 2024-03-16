from lamini_classifier import LaminiClassifier

import csv
import jsonlines

import logging


def main():
    logging.basicConfig(level=logging.DEBUG)

    classifier = LaminiClassifier(augmented_example_count=20)

    examples = load_examples()

    add_data(classifier, examples)

    prompts = {
        "incorrect": "You are a financial analyst with extensive experience at Goldman Sachs. "
        "You are writing a list of questions that you have heard from a client about a specific earnings call. "
        "The question asks about specific numbers mentioned in the call. "
        "Questions, answers, and explanations are short, less than 100 words. "
        "What is an incorrect but plausible answer to the following question?",
        "correct": "You are a financial analyst with extensive experience at Goldman Sachs. "
        "You are writing a list of questions that you have heard from a client about a specific earnings call. "
        "The question asks about specific numbers mentioned in the call. "
        "Questions, answers, and explanations are short, less than 100 words. "
        "You are writing a list of frequently asked questions that you have heard from clients along with correct answers that you have provided.",
    }

    classifier.prompt_train(prompts)

    classifier.save("models/classifier.lamini")


def load_examples():
    path = "/app/lamini-earnings-classify/data/answers.jsonl"

    with jsonlines.open(path) as reader:
        for example in reader:
            yield example


def add_data(classifier, examples):
    positive_examples = []
    negative_examples = []
    for example in examples:
        if "label" in example:
            text = form_prompt(example)
            if example["label"] == 1:
                positive_examples.append(text)
            else:
                negative_examples.append(text)

    if len(positive_examples) > 0:
        classifier.add_data_to_class("correct", positive_examples)

    if len(negative_examples) > 0:
        classifier.add_data_to_class("incorrect", negative_examples)


def form_prompt(example):
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


main()

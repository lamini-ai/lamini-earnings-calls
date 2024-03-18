import jsonlines
from tqdm import tqdm


import logging


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    earnings_calls = load_earnings_calls()

    filtered_calls = filter_calls(earnings_calls)

    save_filtered_calls(filtered_calls)

def load_earnings_calls():
    path = "/app/lamini-extract-numbers/data/earnings-calls-with-numbers.jsonl"

    with jsonlines.open(path) as reader:
        for line in reader:
            logger.info(f"Loaded earnings call for {line['ticker']}")
            yield line

def filter_calls(earnings_calls):
    for call in earnings_calls:
        if call["units"] == "percent":
            if call["value"] < 1.0:
                call["has_value"] = False

        if call["has_value"]:
            yield call

def save_filtered_calls(filtered_calls):
    path = "/app/lamini-extract-numbers/data/earnings-calls-filtered.jsonl"

    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving calls", unit=" calls")
        for call in filtered_calls:
            writer.write(call)
            pbar.update()


main()


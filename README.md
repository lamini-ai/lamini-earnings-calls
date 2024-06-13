# This repo has been deprecated, please refer to https://github.com/lamini-ai/lamini-sdk

# lamini-earnings-calls

Generate a question answer dataset from any earnings call transcript.  Perfect for finetuning LLMs that are experts in current financial data.

Clone this repo

```bash
git clone git@github.com:lamini-ai/lamini-earnings-calls.git
```

Get your [lamini API key](https://app.lamini.ai/account)

Run the script

```bash
./generate-qa.sh
```

# Add your own data

This script reads from [data/earnings-transcripts.jsonl](data/earnings-transcripts.jsonl) .  You can add your own in a format like this:

```json
{
  "date": "Aug 27, 2020, 9:00 p.m. ET",
  "exchange": "NASDAQ: BILI",
  "q": "2020-Q2",
  "ticker": "BILI",
  "transcript": "..."
}
{
  "date": "Jul 30, 2020, 4:30 p.m. ET",
  "exchange": "NYSE: GFF",
  "q": "2020-Q3",
  "ticker": "GFF",
  "transcript": "..."
}
```

You can also modify the data loading function in [lamini_earnings_calls/main.py](lamini_earnings_calls/main.py) to fit your needs:

```python
async def load_earnings_calls():
    path = "/app/lamini-earnings-calls/data/earnings-transcripts.jsonl"

    with jsonlines.open(path) as reader:
        for line in reader:
            logger.info(f"Loaded earnings call for {line['ticker']}")
            yield PromptObject(prompt="", data=line)
```


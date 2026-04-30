import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR,
    f"ner_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


logger = logging.getLogger("ner_benchmark")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.FileHandler(log_file)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

handler.setFormatter(formatter)

logger.addHandler(handler)


MODELS = [
    "dslim/bert-base-NER",
    "StanfordAIMI/stanford-deidentifier-base",
    "Jean-Baptiste/roberta-large-ner-english",
    "Davlan/xlm-roberta-base-ner-hrl",
]

TEST_CASES = [
    {
        "text": (
            "OpenAI signed a contract with Microsoft in Seattle "
            "for 10 million dollars."
        ),
        "expected": {
            "OpenAI",
            "Microsoft",
            "Seattle",
        },
    },
    {
        "text": (
            "Elon Musk visited Berlin and met with executives "
            "from Tesla."
        ),
        "expected": {
            "Elon Musk",
            "Berlin",
            "Tesla",
        },
    },
    {
        "text": (
            "Apple presented new products in California "
            "during WWDC."
        ),
        "expected": {
            "Apple",
            "California",
            "WWDC",
        },
    },
    {
        "text": (
            "Google acquired a startup in London "
            "for 2 billion dollars."
        ),
        "expected": {
            "Google",
            "London",
        },
    },
    {
        "text": (
            "Amazon opened a new office in New York "
            "with support from AWS."
        ),
        "expected": {
            "Amazon",
            "New York",
            "AWS",
        },
    },

    {
        "text": (
            "Илон Маск посетил Москву и встретился с представителями "
            "Роскосмоса."
        ),
        "expected": {
            "Илон Маск",
            "Москву",
            "Роскосмоса",
        },
    },
    {
        "text": (
            "Сбербанк открыл новый офис в Санкт-Петербурге."
        ),
        "expected": {
            "Сбербанк",
            "Санкт-Петербурге",
        },
    },
    {
        "text": (
            "Президент Владимир Путин провёл встречу "
            "в Кремле."
        ),
        "expected": {
            "Владимир Путин",
            "Кремле",
        },
    },
    {
        "text": (
            "Компания Яндекс представила новый сервис "
            "в Казани."
        ),
        "expected": {
            "Яндекс",
            "Казани",
        },
    },
    {
        "text": (
            "Инженеры SpaceX прибыли в Байконур "
            "для подготовки миссии."
        ),
        "expected": {
            "SpaceX",
            "Байконур",
        },
    },
]


def normalize_entities(results: list[dict]) -> set[str]:
    entities = set()

    for item in results:
        word = item["word"].replace("##", "").strip()

        if len(word) > 1:
            entities.add(word)

    return entities


def load_ner_model(model_id: str):
    logger.info(
        "model_loading_started | model=%s",
        model_id,
    )

    start = time.perf_counter()

    ner = pipeline(
        task="ner",
        model=model_id,
        tokenizer=model_id,
        token=HF_TOKEN,
        aggregation_strategy="simple",
    )

    duration = time.perf_counter() - start

    logger.info(
        "model_loaded | model=%s | load_sec=%.4f",
        model_id,
        duration,
    )

    return ner, duration


def benchmark_model(model_id: str) -> dict:
    ner, load_time = load_ner_model(model_id)

    total_expected = 0
    total_found = 0

    inference_times = []

    for index, case in enumerate(TEST_CASES, start=1):
        text = case["text"]
        expected = case["expected"]

        start = time.perf_counter()

        result = ner(text)

        latency = time.perf_counter() - start

        predicted = normalize_entities(result)

        found = 0

        for expected_entity in expected:
            for predicted_entity in predicted:
                if expected_entity.lower() in predicted_entity.lower():
                    found += 1
                    break

        total_expected += len(expected)
        total_found += found

        inference_times.append(latency)

        logger.info(
            "case | model=%s | idx=%s | expected=%s | predicted=%s | found=%s/%s | latency=%.4f",
            model_id,
            index,
            list(expected),
            list(predicted),
            found,
            len(expected),
            latency,
        )

    recall = total_found / total_expected

    avg_latency = sum(inference_times) / len(inference_times)

    summary = {
        "model": model_id,
        "recall": recall,
        "found": total_found,
        "expected": total_expected,
        "load_sec": load_time,
        "avg_inf_sec": avg_latency,
    }

    logger.info(
        "model_summary | model=%s | recall=%.4f | found=%s/%s | load_sec=%.4f | avg_inf_sec=%.4f",
        model_id,
        recall,
        total_found,
        total_expected,
        load_time,
        avg_latency,
    )

    return summary

def log_summary_table(rows: list[dict]) -> None:
    if not rows:
        logger.warning("summary_empty")
        return

    headers = [
        "MODEL",
        "RECALL",
        "FOUND",
        "LOAD_SEC",
        "AVG_INF_SEC",
    ]

    table_rows = [
        [
            row["model"],
            f'{row["recall"]:.4f}',
            f'{row["found"]}/{row["expected"]}',
            f'{row["load_sec"]:.4f}',
            f'{row["avg_inf_sec"]:.4f}',
        ]
        for row in rows
    ]

    widths = [
        max(len(str(item)) for item in [header] + [row[i] for row in table_rows])
        for i, header in enumerate(headers)
    ]

    def format_row(row):
        return " | ".join(
            str(value).ljust(widths[i])
            for i, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)

    logger.info("summary_table_start")
    logger.info(separator)
    logger.info(format_row(headers))
    logger.info(separator)

    for row in table_rows:
        logger.info(format_row(row))

    logger.info("summary_table_end")

def main():
    logger.info(
        "benchmark_started | models=%s",
        len(MODELS),
    )

    results = []

    for model_id in MODELS:
        try:
            results.append(
                benchmark_model(model_id)
            )
        except Exception as exc:
            logger.exception(
                "benchmark_failed | model=%s | error=%s",
                model_id,
                exc,
            )

    log_summary_table(results)

    logger.info("benchmark_finished | models=%s", len(results))


if __name__ == "__main__":
    main()
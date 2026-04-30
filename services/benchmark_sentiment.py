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
    f"sentiment_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)


logger = logging.getLogger("sentiment_benchmark")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_file)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


MODELS = [
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "ProsusAI/finbert",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "nlptown/bert-base-multilingual-uncased-sentiment",
]


TEST_DATA = [
    # English
    ("The product is excellent, I am very happy with the purchase.", "positive"),
    ("Terrible quality, it broke after two days.", "negative"),
    ("The delivery was fast and the packaging was good.", "positive"),
    ("Customer support was rude and completely useless.", "negative"),
    ("The interface is simple, clean and pleasant to use.", "positive"),
    ("The application crashes constantly and loses my data.", "negative"),
    ("This is the best service I have used this year.", "positive"),
    ("The price is too high for such poor quality.", "negative"),
    ("Everything works as expected, no complaints.", "positive"),
    ("I regret buying this product.", "negative"),
    ("The company reported strong revenue growth this quarter.", "positive"),
    ("The company warned about declining profits and weak demand.", "negative"),

    # Russian
    ("Отличный продукт, полностью доволен покупкой.", "positive"),
    ("Качество ужасное, сломалось через два дня.", "negative"),
    ("Доставка была быстрой, упаковка аккуратная.", "positive"),
    ("Поддержка отвечает грубо и не помогает.", "negative"),
    ("Интерфейс удобный, современный и понятный.", "positive"),
    ("Приложение постоянно зависает и теряет данные.", "negative"),
    ("Лучший сервис, которым я пользовался в этом году.", "positive"),
    ("Цена слишком высокая для такого качества.", "negative"),
    ("Всё работает стабильно, претензий нет.", "positive"),
    ("Очень разочарован этой покупкой.", "negative"),
]


def normalize_label(model_id: str, raw_label: str) -> str:
    label = raw_label.lower()

    if model_id == "cardiffnlp/twitter-roberta-base-sentiment-latest":
        mapping = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }
        return mapping.get(label, label)

    if model_id == "nlptown/bert-base-multilingual-uncased-sentiment":
        if label in ("1 star", "2 stars"):
            return "negative"

        if label == "3 stars":
            return "neutral"

        if label in ("4 stars", "5 stars"):
            return "positive"

    if "positive" in label or label == "pos":
        return "positive"

    if "negative" in label or label == "neg":
        return "negative"

    if "neutral" in label:
        return "neutral"

    return label


def load_classifier(model_id: str):
    logger.info(
        "model_loading_started | model=%s",
        model_id,
    )

    load_start = time.perf_counter()

    classifier = pipeline(
        task="text-classification",
        model=model_id,
        tokenizer=model_id,
        token=HF_TOKEN,
    )

    load_duration = time.perf_counter() - load_start

    logger.info(
        "model_loaded | model=%s | load_time_sec=%.4f",
        model_id,
        load_duration,
    )

    return classifier, load_duration


def benchmark_model(model_id: str) -> dict:
    classifier, load_time = load_classifier(model_id)

    total = len(TEST_DATA)
    correct = 0

    inference_times = []

    for index, (text, expected) in enumerate(TEST_DATA, start=1):
        infer_start = time.perf_counter()

        result = classifier(
            text,
            truncation=True,
        )[0]

        infer_duration = time.perf_counter() - infer_start

        raw_label = result["label"]
        score = result["score"]

        predicted = normalize_label(
            model_id=model_id,
            raw_label=raw_label,
        )

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        inference_times.append(infer_duration)

        logger.info(
            "test_case | model=%s | idx=%s | expected=%s | predicted=%s | raw=%s | score=%.4f | latency_sec=%.4f | correct=%s",
            model_id,
            index,
            expected,
            predicted,
            raw_label,
            score,
            infer_duration,
            is_correct,
        )

    accuracy = correct / total

    total_inference = sum(inference_times)
    avg_inference = total_inference / total

    summary = {
        "model": model_id,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "load_sec": load_time,
        "total_inference_sec": total_inference,
        "avg_inference_sec": avg_inference,
    }

    logger.info(
        "model_summary | model=%s | accuracy=%.4f | correct=%s/%s | load_sec=%.4f | total_inf_sec=%.4f | avg_inf_sec=%.4f",
        model_id,
        accuracy,
        correct,
        total,
        load_time,
        total_inference,
        avg_inference,
    )

    return summary


def log_summary_table(rows: list[dict]) -> None:
    if not rows:
        logger.warning("summary_empty")
        return

    headers = [
        "MODEL",
        "ACCURACY",
        "CORRECT",
        "LOAD_SEC",
        "TOTAL_INF_SEC",
        "AVG_INF_SEC",
    ]

    table_rows = [
        [
            row["model"],
            f'{row["accuracy"]:.4f}',
            f'{row["correct"]}/{row["total"]}',
            f'{row["load_sec"]:.4f}',
            f'{row["total_inference_sec"]:.4f}',
            f'{row["avg_inference_sec"]:.4f}',
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
        "benchmark_started | hf_token_loaded=%s | models=%s",
        bool(HF_TOKEN),
        len(MODELS),
    )

    results = []

    for model_id in MODELS:
        try:
            result = benchmark_model(model_id)
            results.append(result)

        except Exception as exc:
            logger.exception(
                "benchmark_failed | model=%s | error=%s",
                model_id,
                exc,
            )

    log_summary_table(results)

    logger.info(
        "benchmark_finished | tested_models=%s",
        len(results),
    )


if __name__ == "__main__":
    main()
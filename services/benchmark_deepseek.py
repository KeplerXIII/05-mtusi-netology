import json
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env")


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR,
    f"deepseek_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logger = logging.getLogger("deepseek_benchmark")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.FileHandler(log_file)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logger.addHandler(handler)


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=60,
)


SENTIMENT_TEST_DATA = [
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


NER_TEST_CASES = [
    {
        "text": "OpenAI signed a contract with Microsoft in Seattle for 10 million dollars.",
        "expected": {"OpenAI", "Microsoft", "Seattle"},
    },
    {
        "text": "Elon Musk visited Berlin and met with executives from Tesla.",
        "expected": {"Elon Musk", "Berlin", "Tesla"},
    },
    {
        "text": "Apple presented new products in California during WWDC.",
        "expected": {"Apple", "California", "WWDC"},
    },
    {
        "text": "Google acquired a startup in London for 2 billion dollars.",
        "expected": {"Google", "London"},
    },
    {
        "text": "Amazon opened a new office in New York with support from AWS.",
        "expected": {"Amazon", "New York", "AWS"},
    },
    {
        "text": "Илон Маск посетил Москву и встретился с представителями Роскосмоса.",
        "expected": {"Илон Маск", "Москву", "Роскосмоса"},
    },
    {
        "text": "Сбербанк открыл новый офис в Санкт-Петербурге.",
        "expected": {"Сбербанк", "Санкт-Петербурге"},
    },
    {
        "text": "Президент Владимир Путин провёл встречу в Кремле.",
        "expected": {"Владимир Путин", "Кремле"},
    },
    {
        "text": "Компания Яндекс представила новый сервис в Казани.",
        "expected": {"Яндекс", "Казани"},
    },
    {
        "text": "Инженеры SpaceX прибыли в Байконур для подготовки миссии.",
        "expected": {"SpaceX", "Байконур"},
    },
]


def extract_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise

        return json.loads(content[start:end + 1])


def ask_json(system_prompt: str, user_prompt: str) -> tuple[dict, float, dict]:
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    latency = time.perf_counter() - start
    content = response.choices[0].message.content
    data = extract_json(content)

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return data, latency, usage


def normalize_label(label: str) -> str:
    value = label.strip().lower()

    if value in {"positive", "pos", "позитивная", "положительная"}:
        return "positive"

    if value in {"negative", "neg", "негативная", "отрицательная"}:
        return "negative"

    if value in {"neutral", "нейтральная"}:
        return "neutral"

    return value


def benchmark_sentiment() -> dict:
    logger.info(
        "sentiment_started | model=%s | cases=%s",
        DEEPSEEK_MODEL,
        len(SENTIMENT_TEST_DATA),
    )

    correct = 0
    inference_times = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    system_prompt = (
        "You are a strict sentiment classifier. "
        "Return only valid JSON. "
        "Classify text sentiment as positive, negative, or neutral."
    )

    for index, (text, expected) in enumerate(SENTIMENT_TEST_DATA, start=1):
        user_prompt = (
            'Return JSON in this schema: {"label": "positive|negative|neutral"}.\n'
            f"Text: {text}"
        )

        try:
            data, latency, usage = ask_json(system_prompt, user_prompt)
            predicted = normalize_label(str(data.get("label", "")))
            is_correct = predicted == expected

            correct += int(is_correct)
            inference_times.append(latency)

            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)

            logger.info(
                "sentiment_case | model=%s | idx=%s | expected=%s | predicted=%s | latency_sec=%.4f | correct=%s | usage=%s | text=%s",
                DEEPSEEK_MODEL,
                index,
                expected,
                predicted,
                latency,
                is_correct,
                usage,
                text,
            )

        except Exception as exc:
            logger.exception(
                "sentiment_case_failed | idx=%s | expected=%s | error=%s | text=%s",
                index,
                expected,
                exc,
                text,
            )

    total = len(SENTIMENT_TEST_DATA)
    accuracy = correct / total
    total_inference = sum(inference_times)
    avg_inference = total_inference / len(inference_times) if inference_times else 0

    summary = {
        "task": "sentiment",
        "model": DEEPSEEK_MODEL,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "total_inference_sec": total_inference,
        "avg_inference_sec": avg_inference,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }

    logger.info(
        "sentiment_summary | model=%s | accuracy=%.4f | correct=%s/%s | total_inf_sec=%.4f | avg_inf_sec=%.4f | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
        DEEPSEEK_MODEL,
        accuracy,
        correct,
        total,
        total_inference,
        avg_inference,
        total_prompt_tokens,
        total_completion_tokens,
        total_tokens,
    )

    return summary


def normalize_entities(values) -> set[str]:
    if not values:
        return set()

    result = set()

    for value in values:
        text = str(value).strip()

        if text:
            result.add(text)

    return result


def benchmark_ner() -> dict:
    logger.info(
        "ner_started | model=%s | cases=%s",
        DEEPSEEK_MODEL,
        len(NER_TEST_CASES),
    )

    total_expected = 0
    total_found = 0
    inference_times = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    system_prompt = (
        "You are a strict named entity recognition system. "
        "Extract only named entities explicitly present in the text. "
        "Return only valid JSON."
    )

    for index, case in enumerate(NER_TEST_CASES, start=1):
        text = case["text"]
        expected = case["expected"]

        total_expected += len(expected)

        user_prompt = (
            'Return JSON in this schema: {"entities": ["entity1", "entity2"]}.\n'
            "Extract people, organizations, companies, locations, and events.\n"
            "Do not translate entities. Preserve original spelling and word forms.\n"
            f"Text: {text}"
        )

        try:
            data, latency, usage = ask_json(system_prompt, user_prompt)
            predicted = normalize_entities(data.get("entities", []))

            found = 0

            for expected_entity in expected:
                for predicted_entity in predicted:
                    if expected_entity.lower() in predicted_entity.lower():
                        found += 1
                        break

            total_found += found
            inference_times.append(latency)

            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)

            logger.info(
                "ner_case | model=%s | idx=%s | expected=%s | predicted=%s | found=%s/%s | latency_sec=%.4f | usage=%s | text=%s",
                DEEPSEEK_MODEL,
                index,
                list(expected),
                list(predicted),
                found,
                len(expected),
                latency,
                usage,
                text,
            )

        except Exception as exc:
            logger.exception(
                "ner_case_failed | idx=%s | error=%s | text=%s",
                index,
                exc,
                text,
            )

    recall = total_found / total_expected if total_expected else 0
    total_inference = sum(inference_times)
    avg_inference = total_inference / len(inference_times) if inference_times else 0

    summary = {
        "task": "ner",
        "model": DEEPSEEK_MODEL,
        "recall": recall,
        "found": total_found,
        "expected": total_expected,
        "total_inference_sec": total_inference,
        "avg_inference_sec": avg_inference,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }

    logger.info(
        "ner_summary | model=%s | recall=%.4f | found=%s/%s | total_inf_sec=%.4f | avg_inf_sec=%.4f | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
        DEEPSEEK_MODEL,
        recall,
        total_found,
        total_expected,
        total_inference,
        avg_inference,
        total_prompt_tokens,
        total_completion_tokens,
        total_tokens,
    )

    return summary


def log_summary_table(rows: list[dict]) -> None:
    headers = [
        "TASK",
        "MODEL",
        "METRIC",
        "SCORE",
        "RESULT",
        "TOTAL_INF_SEC",
        "AVG_INF_SEC",
        "TOKENS",
    ]

    table_rows = []

    for row in rows:
        if row["task"] == "sentiment":
            metric = "ACCURACY"
            score = f'{row["accuracy"]:.4f}'
            result = f'{row["correct"]}/{row["total"]}'
        else:
            metric = "RECALL"
            score = f'{row["recall"]:.4f}'
            result = f'{row["found"]}/{row["expected"]}'

        table_rows.append(
            [
                row["task"],
                row["model"],
                metric,
                score,
                result,
                f'{row["total_inference_sec"]:.4f}',
                f'{row["avg_inference_sec"]:.4f}',
                str(row["total_tokens"]),
            ]
        )

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
        "benchmark_started | provider=deepseek | base_url=%s | model=%s",
        DEEPSEEK_BASE_URL,
        DEEPSEEK_MODEL,
    )

    results = [
        benchmark_sentiment(),
        benchmark_ner(),
    ]

    log_summary_table(results)

    logger.info("benchmark_finished | tasks=%s", len(results))


if __name__ == "__main__":
    main()
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from dotenv import load_dotenv
from huggingface_hub import HfApi


load_dotenv()

# === ЛОГИ ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR,
    f"hf_model_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
    ],
)

logger = logging.getLogger("hf_model_search")

logger.info("logger_initialized | file=%s", log_file)

# === HF API ===
api = HfApi(token=os.getenv("HF_TOKEN"))


@dataclass
class TaskConfig:
    name: str
    pipeline_tag: str
    limit: int = 10
    required_tags: Optional[Tuple[str, ...]] = None


TASKS = [
    TaskConfig(
        name="sentiment",
        pipeline_tag="text-classification",
        limit=20,
        required_tags=("sentiment",),
    ),
    TaskConfig(
        name="ner",
        pipeline_tag="token-classification",
        limit=20,
        required_tags=("ner",),
    ),
]


def get_license(info) -> str:
    if not info.card_data:
        return "unknown"

    try:
        return info.card_data.get("license", "unknown")
    except Exception:
        return "unknown"


def short_tags(tags, limit: int = 6) -> str:
    if not tags:
        return "-"
    return ", ".join(tags[:limit])


def collect_models(task: TaskConfig) -> list[dict]:
    logger.info(
        "search_started | task=%s | pipeline_tag=%s | limit=%s",
        task.name,
        task.pipeline_tag,
        task.limit,
    )

    models = api.list_models(
        pipeline_tag=task.pipeline_tag,
        sort="downloads",
        limit=task.limit,
    )

    result = []

    for model in models:
        try:
            info = api.model_info(model.id)

            row = {
                "task": task.name,
                "model_id": info.id,
                "author": info.author or "-",
                "downloads": info.downloads or 0,
                "likes": info.likes or 0,
                "license": get_license(info),
                "tags": short_tags(info.tags),
            }

            result.append(row)

            logger.info(
                "model | task=%s | id=%s | downloads=%s | likes=%s | license=%s",
                row["task"],
                row["model_id"],
                row["downloads"],
                row["likes"],
                row["license"],
            )

        except Exception as exc:
            logger.exception(
                "model_info_failed | task=%s | model=%s | error=%s",
                task.name,
                model.id,
                exc,
            )

    logger.info(
        "search_finished | task=%s | collected=%s",
        task.name,
        len(result),
    )

    return result


def log_summary_table(rows: list[dict]) -> None:
    if not rows:
        logger.warning("summary_empty")
        return

    headers = ["TASK", "MODEL", "DOWNLOADS", "LIKES", "LICENSE"]

    table_rows = [
        [
            row["task"],
            row["model_id"],
            f'{row["downloads"]:,}',
            f'{row["likes"]:,}',
            row["license"],
        ]
        for row in rows
    ]

    widths = [
        max(len(str(item)) for item in [header] + [row[i] for row in table_rows])
        for i, header in enumerate(headers)
    ]

    def format_row(row):
        return " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))

    separator = "-+-".join("-" * w for w in widths)

    logger.info("summary_table_start")
    logger.info(separator)
    logger.info(format_row(headers))
    logger.info(separator)

    for row in table_rows:
        logger.info(format_row(row))

    logger.info("summary_table_end")


def main():
    all_rows = []

    for task in TASKS:
        rows = collect_models(task)
        all_rows.extend(rows)

    log_summary_table(all_rows)

    logger.info("completed | total_models=%s", len(all_rows))


if __name__ == "__main__":
    main()
import json
import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def ask_json(system_prompt: str, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    return json.loads(content)


def sentiment_demo(text: str):
    if not text.strip():
        return "Введите текст"

    data = ask_json(
        system_prompt=(
            "You are a strict sentiment classifier. "
            "Return only valid JSON with field label. "
            "Allowed labels: positive, negative, neutral."
        ),
        user_prompt=(
            'Return JSON in this schema: {"label": "positive|negative|neutral"}.\n'
            f"Text: {text}"
        ),
    )

    return data.get("label", "unknown")


def ner_demo(text: str):
    if not text.strip():
        return []

    data = ask_json(
        system_prompt=(
            "You are a strict named entity recognition system. "
            "Extract only named entities explicitly present in the text. "
            "Return only valid JSON."
        ),
        user_prompt=(
            'Return JSON in this schema: {"entities": ["entity1", "entity2"]}.\n'
            "Extract people, organizations, companies, locations, and events.\n"
            "Preserve original spelling and word forms.\n"
            f"Text: {text}"
        ),
    )

    entities = data.get("entities", [])

    return [[entity] for entity in entities]


with gr.Blocks(title="DeepSeek NLP Demo") as demo:
    gr.Markdown("# DeepSeek NLP Demo")
    gr.Markdown("Демо для классификации тональности и извлечения сущностей.")

    with gr.Tab("Sentiment Analysis"):
        sentiment_input = gr.Textbox(
            label="Текст",
            lines=5,
            value="Отличный продукт, полностью доволен покупкой.",
        )
        sentiment_button = gr.Button("Определить тональность")
        sentiment_output = gr.Textbox(label="Результат")

        sentiment_button.click(
            fn=sentiment_demo,
            inputs=sentiment_input,
            outputs=sentiment_output,
        )

    with gr.Tab("Named Entity Recognition"):
        ner_input = gr.Textbox(
            label="Текст",
            lines=5,
            value="Илон Маск посетил Москву и встретился с представителями Роскосмоса.",
        )
        ner_button = gr.Button("Извлечь сущности")
        ner_output = gr.Dataframe(
            headers=["Entity"],
            label="Сущности",
        )

        ner_button.click(
            fn=ner_demo,
            inputs=ner_input,
            outputs=ner_output,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
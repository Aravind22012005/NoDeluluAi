from openai import OpenAI
import random
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE

client = OpenAI(api_key=OPENAI_API_KEY)

# cache to speed up repeated calls
cache = {}


def call_llm(prompt, temperature=None):

    if temperature is None:
        temperature = TEMPERATURE
    if random.random() < 0.2:
        temperature = 1.7

    key = prompt + str(temperature)

    if key in cache:
        return cache[key]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    text = resp.choices[0].message.content.strip()

    cache[key] = text
    return text

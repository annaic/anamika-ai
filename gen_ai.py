import os

import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
client = AsyncOpenAI(api_key=OPEN_AI_KEY)


@cl.step(type="llm")
async def call_gpt4(message_history):
    settings = {
        "model": "gpt-4",
        "stream": False,
        "temperature": 0.1,
    }
    response = await client.chat.completions.create(
        messages=message_history, **settings,
    )
    message = response.choices[0].message
    if message.content:
        cl.context.current_step.output = message.content
    return message

import os

import chainlit as cl
from chainlit.playground.providers.openai import stringify_function_call
from dotenv import load_dotenv

from app import call_tool
from openai_wrapper import AsyncCustomOpenAIClient

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
client = AsyncCustomOpenAIClient(api_key=OPEN_AI_KEY, base_url="http://localhost:8000/v1")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Austin, TX"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]
@cl.step(type="llm")
async def call_function_ai(message_history):
    settings = {
        "model": "gpt-4",
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "temperature": 0.1,
    }
    response = await client.chat.completions.create(
        messages=message_history, **settings,
    )
    message = response.choices[0].message
    for tool_call in message.tool_calls or []:
        if tool_call.type == "function":
            await call_tool(tool_call, message_history)

    if message.content:
        cl.context.current_step.output = message.content
    elif message.tool_calls:
        completion = stringify_function_call(message.tool_calls[0].function)

        cl.context.current_step.language = "json"
        cl.context.current_step.output = completion

    return message
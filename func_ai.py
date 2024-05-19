import ast
import json

import chainlit as cl
from chainlit.playground.providers.openai import stringify_function_call

from openai_wrapper import AsyncCustomOpenAIClient

OPEN_AI_KEY = "Fake_key"
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


@cl.step(type="tool")
async def call_tool(tool_call, message_history):
    """
    :param tool_call: The tool call object representing the current tool being executed.
    :type tool_call: ToolCall
    :param message_history: The list of messages exchanged between the user and the tool.
    :type message_history: List[Message]
    :return: None

    This method is a step decorator for executing a tool. It takes a `tool_call` object of type `ToolCall` and a `message_history` list of type `List[Message]` as parameters. The method
    * performs the following steps:

    1. Extracts the function name and arguments from the `tool_call` object.
    2. Sets the current step's name to the extracted function name.
    3. Sets the current step's input to the extracted arguments.
    4. Calls the `get_current_weather` function with the specified location and unit arguments.
    5. Sets the current step's output to the response obtained from calling `get_current_weather`.
    6. Sets the current step's language to "json".
    7. Appends a new message object to the `message_history` list, representing the function call and its response.

    Note: This method does not return anything.
    """
    function_name = tool_call.function.name
    arguments = ast.literal_eval(tool_call.function.arguments)

    current_step = cl.context.current_step
    current_step.name = function_name

    current_step.input = arguments

    function_response = get_current_weather(
        location=arguments.get("location"),
        unit=arguments.get("unit"),
    )

    current_step.output = function_response
    current_step.language = "json"

    message_history.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
            "tool_call_id": tool_call.id,
        }
    )


@cl.step(type="llm")
async def call_function_ai(message_history):
    settings = {
        "model": "gpt-4",  # This is actually gorilla that we are running locally, keeping it as gpt-4 for fallback
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


def get_current_weather(location: str, unit: str):
    """Get the current weather in a given location"""
    unit = unit or "Fahrenheit"
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }

    return json.dumps(weather_info)

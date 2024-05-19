import ast
import json
import os
from openai import AsyncOpenAI
from chainlit.playground.providers.openai import stringify_function_call
from openai_wrapper import AsyncCustomOpenAIClient
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
# client = AsyncCustomOpenAIClient(api_key=OPEN_AI_KEY, base_url="http://localhost:8000/v1")
# client = AsyncOpenAI(api_key=OPEN_AI_KEY)
client = AsyncCustomOpenAIClient(api_key=OPEN_AI_KEY)

MAX_ITER = 5

cl.instrument_openai()


# Example dummy function hard-coded to return the same weather.
# TODO: Change this to a real implementation
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


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{
            "role": "system",
            "content": "Your name is Goril an artificial intelligence specialized in selecting tools to resolve user "
                       "requests. If the user's request does not require a tool, then maintain a friendly and fluid "
                       "conversation with the user. If parameters are missing to run a tool, notify it and suggest a "
                       "solution. Additionally, you must be attentive to the language the user speaks to respond in "
                       "the same language."
        }],
    )


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
async def call_gpt4(message_history):
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


@cl.on_message
async def run_conversation(message: cl.Message):
    """
    This function is called everytime a user inputs a message in the UI.
    It sends back an intermediate response from the tool followed by the final answer
    Args:
        message: The user's message
    Returns:
        None
    """
    message_history = cl.user_session.get("message_history")
    message_history.append({
        "name": "user",
        "role": "user",
        "content": message.content
    })
    curr_iter = 0
    while curr_iter < MAX_ITER:
        message = await call_gpt4(message_history)
        if not message.tool_calls:
            await cl.Message(content=message.content, author="Answer").send()
            break
        curr_iter += 1

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

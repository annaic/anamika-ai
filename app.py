import chainlit as cl

from func_ai import call_function_ai
from gen_ai import call_gpt4

MAX_ITER = 5

cl.instrument_openai()


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

    message = await call_function_ai(message_history)
    if not message.tool_calls:
        # Use generative AI to format the response from the function call if it worked.
        message = await call_gpt4(message_history)
        await cl.Message(content=message.content, author="Answer").send()
    else:
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

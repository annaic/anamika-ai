import chainlit as cl


@cl.step
def tool():
    return "Response from the tool!"


@cl.on_message
async def main(message: cl.Message):
    """
    This function is called everytime a user inputs a message in the UI.
    It sends back an intermediate response from the tool followed by the final answer
    Args:
        message: The user's message
    Returns:
        None
    """
    # Call the tool
    tool()

    # Send the final answer.
    await cl.Message(content="This is the final answer").send()

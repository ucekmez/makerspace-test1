import os
import chainlit as cl
from openai import OpenAI
from loguru import logger

# Initialize OpenAI client with error handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.info("Warning: OPENAI_API_KEY not found in environment variables. Please set it before using the application.")

client = OpenAI(api_key=api_key)

# Configure Chainlit settings
os.environ["CHAINLIT_AUTH_SECRET"] = "secret"
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"
os.environ["CHAINLIT_HOST"] = "0.0.0.0"
os.environ["CHAINLIT_PORT"] = "7860"

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session when a user starts a new chat.
    """
    logger.info("DEBUG: Chat session started")
    # Set the initial system message to guide the AI assistant's behavior
    system_message = {"role": "system", "content": "You are a highly humorous, witty, and friendly AI assistant with an endless supply of jokes, puns, and playful sarcasm. Your personality is lighthearted, fun, and engaging, making every conversation feel like chatting with a hilarious best friend."}
    
    # Store the conversation history in the user session
    welcome_msg = cl.Message(content="Hello! What can I help you with today?")
    await welcome_msg.send()
    logger.info("DEBUG: Welcome message sent")
    
    # Initialize chat history with the system message
    cl.user_session.set("message_history", [system_message])
    logger.info("DEBUG: Message history initialized with system message")

@cl.on_message
async def on_message(message: cl.Message):
    """
    Process incoming user messages and generate AI responses.
    
    Args:
        message: The user's message object
    """
    logger.info(f"DEBUG: Received user message: {message.content}")
    
    # Get message history from the user session
    message_history = cl.user_session.get("message_history")
    logger.info(f"DEBUG: Retrieved message history")
    
    # Add the user message to history
    message_history.append({"role": "user", "content": message.content})
    
    try:
        # Create a new message for streaming response
        msg = cl.Message(content="")
        await msg.send()
        
        # Call the OpenAI API with the complete message history
        logger.info("DEBUG: Calling OpenAI API with message history")
        
        full_response = ""
        
        # Stream the response
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_history,
            temperature=0.7,
            max_tokens=800,
            stream=True
        )
        
        logger.info("DEBUG: Started streaming response from OpenAI")
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                await msg.stream_token(content)
                logger.info(f"DEBUG: Streamed token: {content}")
                
        logger.info(f"DEBUG: Completed streaming response")
        
        # Update the message with the final response content
        await msg.update()
        logger.info(f"DEBUG: Message updated with final content")
        
        # Add the assistant's response to the message history
        message_history.append({"role": "assistant", "content": full_response})
        
        # Update the message history in the user session
        cl.user_session.set("message_history", message_history)
        logger.info(f"DEBUG: Updated message history, new length: {len(message_history)}")
        
    except Exception as e:
        # Handle errors gracefully
        logger.info(f"DEBUG: ERROR encountered: {str(e)}")
        error_message = f"Sorry, I encountered an error: {str(e)}"
        await cl.Message(error_message).send()
        logger.info("DEBUG: Error message displayed to user") 
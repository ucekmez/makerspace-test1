import os
import uuid 
from datetime import datetime
import chainlit as cl
from openai import OpenAI
from loguru import logger
# Import custom DB utils
from db_utils import init_db, get_sessions, load_session, save_session 

# Initialize the custom database
init_db()

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

# System message definition
SYSTEM_MESSAGE = {"role": "system", "content": "You are a highly humorous, witty, and friendly AI assistant with an endless supply of jokes, puns, and playful sarcasm. Your personality is lighthearted, fun, and engaging, making every conversation feel like chatting with a hilarious best friend."}

# Password Authentication Callback (username only check)
@cl.password_auth_callback
async def auth_callback(credentials) -> cl.User | None:
    # Simplified based on previous findings: credentials is the username string
    logger.info(f"DEBUG: auth_callback received credentials: {credentials} (Type: {type(credentials)})")
    if isinstance(credentials, str) and credentials == "admin":
        logger.info(f"DEBUG: User 'admin' authenticated.")
        return cl.User(identifier="admin") 
    else:
        logger.warning(f"DEBUG: Auth failed for '{credentials}'.")
        return None

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize chat using AskUserMessage for session selection (custom DB).
    """
    logger.info("DEBUG: on_chat_start called (custom DB).")
    app_user = cl.user_session.get("user")
    if not app_user:
        logger.error("ERROR: No authenticated user found in on_chat_start!")
        await cl.ErrorMessage("Authentication error. Please refresh.").send()
        return
        
    user_id = app_user.identifier
    logger.info(f"DEBUG: User '{user_id}' started chat.")

    # --- Custom Session Selection Logic --- 
    existing_sessions = get_sessions(user_id)
    session_options = {}
    prompt_lines = [
        f"Welcome {user_id}! Please choose an option:",
        "  - Type 'new' to start a new chat."
    ]

    if existing_sessions:
        prompt_lines.append("  - Type the ID of a session below to resume:")
        for i, session in enumerate(existing_sessions[:5]): # Show latest 5
            last_updated_str = datetime.fromisoformat(session['last_updated']).strftime('%Y-%m-%d %H:%M')
            session_id_db = session['session_id']
            prompt_lines.append(f"    - ID: `{session_id_db}` (Last updated: {last_updated_str})") 
            session_options[session_id_db] = session_id_db # Store full ID
    else:
         prompt_lines.append("  (No previous sessions found)")
    
    prompt_string = "\n".join(prompt_lines)

    # Ask user for input
    res = None
    try:
        res = await cl.AskUserMessage(content=prompt_string, timeout=120).send() # Increased timeout
    except Exception as e:
        logger.error(f"Error displaying AskUserMessage: {e}")
        await cl.Message(content="Error displaying options. Starting new chat.").send()

    session_id = None
    message_history = None
    user_input = None

    # Process user input
    if res and res.get("output"):
        user_input = res.get("output").strip()
        logger.info(f"DEBUG: User '{user_id}' entered: '{user_input}'")
        
        if user_input.lower() == "new":
            logger.info(f"DEBUG: User '{user_id}' chose NEW chat.")
            session_id = str(uuid.uuid4())
            message_history = [SYSTEM_MESSAGE]
            save_session(session_id, user_id, message_history) # Save new session
            await cl.Message(content="Starting a fresh chat! Let the fun begin! 😄").send()
            
        elif user_input in session_options: # Check if input matches a valid session ID
            session_id = session_options[user_input]
            logger.info(f"DEBUG: User '{user_id}' chose RESUME session: {session_id}")
            message_history = load_session(session_id, user_id)
            if not message_history: # Load failed or didn't belong to user
                 logger.warning(f"DEBUG: Failed load/auth for session {session_id}. Starting new.")
                 session_id = str(uuid.uuid4())
                 message_history = [SYSTEM_MESSAGE]
                 save_session(session_id, user_id, message_history)
                 await cl.Message(content="Couldn't load that session. Starting a new one instead!").send()
            else:
                 # Successfully loaded history
                 await cl.Message(content=f"Resuming chat (ID: ...{session_id[-6:]})...").send()
                 # Display previous messages on resume
                 for msg in message_history:
                     if msg["role"] != "system":
                         author = user_id if msg["role"] == "user" else "Assistant"
                         # Use author field for clarity
                         await cl.Message(content=msg["content"], author=author).send() 
        else:
             logger.warning(f"DEBUG: Unrecognized input '{user_input}' for user '{user_id}'. Starting new chat.")
             user_input = None # Signal to default logic

    # Default to new chat if no valid selection/error/timeout
    if not session_id:
        logger.info(f"DEBUG: Defaulting to NEW chat for user '{user_id}'.")
        session_id = str(uuid.uuid4())
        message_history = [SYSTEM_MESSAGE]
        save_session(session_id, user_id, message_history)
        if res is None or user_input is None: # Only send if AskUserMessage timed out/failed or input was invalid
             await cl.Message(content="Okay, starting a new chat for you. ✨").send()

    # Store the determined session ID and history in the user session
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("message_history", message_history)
    logger.info(f"DEBUG: Session {session_id} initialized for user '{user_id}'. History length: {len(message_history)}")
    # --- End Custom Session Selection Logic ---

@cl.on_message
async def on_message(message: cl.Message):
    """
    Process messages using custom DB persistence.
    """
    # Retrieve state from user session
    app_user = cl.user_session.get("user")
    session_id = cl.user_session.get("session_id")
    message_history = cl.user_session.get("message_history")

    # Validate state
    if not app_user or not session_id or message_history is None:
         logger.error("ERROR: Missing user/session/history in on_message! Cannot process.")
         await cl.ErrorMessage("Session error or expired. Please refresh to start a new chat.").send()
         return # Stop processing if state is invalid
         
    user_id = app_user.identifier
    logger.info(f"DEBUG: Received message for session {session_id}, user {user_id}.")

    # Add user message to the history list
    message_history.append({"role": "user", "content": message.content})

    try:
        # Send message to OpenAI API
        msg = cl.Message(content="") # Placeholder for streaming
        await msg.send()

        logger.info(f"DEBUG: Calling OpenAI API for session {session_id}...")
        full_response = ""
        stream = client.chat.completions.create(
            model="gpt-4o-mini", messages=message_history, temperature=0.7,
            max_tokens=800, stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                await msg.stream_token(content)
        await msg.update()
        logger.info(f"DEBUG: OpenAI response received for session {session_id}.")

        # Add assistant response to history list
        message_history.append({"role": "assistant", "content": full_response})
        
        # Update message history in the user session state
        cl.user_session.set("message_history", message_history)
        
        # Save the updated history to the custom database
        save_session(session_id, user_id, message_history)
        logger.info(f"DEBUG: Saved history for session {session_id}, user {user_id}. Length: {len(message_history)}")

    except Exception as e:
        logger.exception(f"ERROR processing message for session {session_id}: {e}")
        await cl.Message(content=f"An error occurred: {e}", author="Error Bot").send() 
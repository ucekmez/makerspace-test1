from config import *

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """
You are a helpful assistant. 
Use the following context to answer the user's question accurately. 
If the context doesn't contain the answer, say you don't know. 
Refer to the source documents when possible. Always respond in max 3 sentences.

Context:
{context}

User Question: {question}
Answer:""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)


DEFAULT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": """

**Core Persona:**
You are a highly humorous, witty, and friendly AI assistant. You're brimming with jokes, puns, and playful sarcasm. Your personality is lighthearted, fun, and engaging—imagine chatting with a hilarious best friend.

**Internal Reasoning Process (Crucial: DO NOT reveal this process in your output):**
Before answering any question, you MUST first engage in a hidden, internal step-by-step thought process. Analyze the request, break down the problem (especially for mathematical questions), consider different humorous angles or witty responses, and formulate the most accurate *and* entertaining answer that fits your persona.

**Final Output Requirements:**
*   Present ONLY the final answer. Your internal reasoning steps MUST remain hidden.
*   Be concise: Your entire response must be a maximum of 3 sentences *total*, including any mathematical formulas.
*   Ensure the response is infused with your signature humor, wit, and friendly tone.

**Formatting Instructions:**
*   Use standard Markdown for text formatting (like *italics*, **bold**, lists) when it improves clarity or fits the tone.
*   For mathematical explanations:
    *   Break down steps clearly using plain text and standard symbols (+, -, *, /, =).
    *   You can use inline code formatting (backticks `` ` ``) for simple equations, for example: `12 apples / 4 apples/pack = 3 packs`.
    *   **Strictly avoid LaTeX syntax (NO $, $$, \[, \(, \frac, etc.).** Describe complex fractions or symbols in words if necessary.
""".strip(),
}



@cl.password_auth_callback
async def auth_callback(credentials) -> cl.User | None:
    logger.info(f"DEBUG: auth_callback received: {credentials}")
    if isinstance(credentials, str) and credentials == "admin":
        logger.info("DEBUG: User 'admin' authenticated.")
        return cl.User(identifier="admin")
    else:
        logger.warning(f"DEBUG: Auth failed for '{credentials}'.")
        return None



@cl.on_chat_start
async def on_chat_start():
    """
    Initialize chat using custom DB session selection OR start fresh.
    Instructs user on how to upload a document.
    """
    logger.info("DEBUG: on_chat_start called (custom DB).")
    app_user = cl.user_session.get("user")
    if not app_user:
        logger.error("ERROR: No authenticated user found!")
        return
    user_id = app_user.identifier

    existing_sessions = get_sessions(user_id)
    session_options = {}
    prompt_lines = [
        f"Welcome {user_id}! Choose an option:",
        "- Type 'new' for a new chat.",
    ]
    if existing_sessions:
        prompt_lines.append("- Type a Session ID to resume:")
        for session in existing_sessions[:5]:
            last_updated_str = datetime.fromisoformat(session["last_updated"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            sid = session["session_id"]
            prompt_lines.append(f"  - `{sid}` (Updated: {last_updated_str})")
            session_options[sid] = sid
    else:
        prompt_lines.append("(No previous sessions)")

    prompt_string = "\n".join(prompt_lines)

    res = None
    try:
        res = await cl.AskUserMessage(content=prompt_string, timeout=120).send()
    except Exception as e:
        logger.error(f"Error asking for session: {e}")

    session_id = None
    message_history = None
    user_input = res.get("output").strip() if res and res.get("output") else None

    if user_input and user_input.lower() == "new":
        session_id = str(uuid.uuid4())
        message_history = [DEFAULT_SYSTEM_MESSAGE]
        save_session(session_id, user_id, message_history)
        await cl.Message(
            content="Starting fresh! Ask general questions or type 'upload' to process a PDF."
        ).send()
    elif user_input and user_input in session_options:
        session_id = session_options[user_input]
        message_history = load_session(session_id, user_id)
        if not message_history:
            session_id = str(uuid.uuid4())
            message_history = [DEFAULT_SYSTEM_MESSAGE]
            save_session(session_id, user_id, message_history)
            await cl.Message(
                content="Couldn't load session. Starting new. Type 'upload' to process a PDF."
            ).send()
        else:
            await cl.Message(
                content=f"Resuming chat... Ask questions or type 'upload' to process a PDF."
            ).send()
            for msg in message_history:
                if msg["role"] != "system":
                    author = user_id if msg["role"] == "user" else "Assistant"
                    await cl.Message(content=msg["content"], author=author).send()
    else:
        session_id = str(uuid.uuid4())
        message_history = [DEFAULT_SYSTEM_MESSAGE]
        save_session(session_id, user_id, message_history)
        await cl.Message(
            content="Okay, starting a new chat. Ask questions or type 'upload' to process a PDF."
        ).send()

    cl.user_session.set("session_id", session_id)
    cl.user_session.set("message_history", message_history)
    logger.info(f"DEBUG: Session {session_id} initialized for user '{user_id}'.")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Process messages: Check for 'upload' command, use RAG if active, else default.
    Saves history to custom DB.
    """
    app_user = cl.user_session.get("user")
    session_id = cl.user_session.get("session_id")
    message_history = cl.user_session.get("message_history")

    if message.content.strip().lower() == "upload":
        logger.info(f"DEBUG: Received 'upload' command from user {app_user.identifier}")
        await handle_upload_request()  # Call the upload handler
        return  # Stop further processing for the 'upload' message itself

    rag_chain = cl.user_session.get("rag_chain")

    if not app_user or not session_id or message_history is None:
        logger.error("ERROR: Missing user/session/history in on_message!")
        await cl.ErrorMessage("Session error. Please refresh.").send()
        return

    user_id = app_user.identifier
    logger.info(
        f"DEBUG: Received message for session {session_id}, user {user_id}. RAG active: {rag_chain is not None}"
    )

    message_history.append({"role": "user", "content": message.content})

    try:
        response_message = cl.Message(content="")
        await response_message.send()

        full_response = ""
        text_elements = []

        if rag_chain:
            logger.info(f"DEBUG: Using RAG chain for session {session_id}.")
            cb = cl.AsyncLangchainCallbackHandler(
                stream_final_answer=True, answer_prefix_tokens=["Answer:"]
            )

            langchain_history = []
            history_to_convert = message_history
            if history_to_convert and history_to_convert[0]["role"] == "system":
                history_to_convert = history_to_convert[
                    1:
                ]  # Skip system message for chain history

            for msg_dict in history_to_convert:
                role = msg_dict.get("role")
                content = msg_dict.get("content", "")
                if role == "user":
                    langchain_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_history.append(AIMessage(content=content))

            chain_input = {
                "question": message.content,
                "chat_history": langchain_history,
            }

            res = await rag_chain.acall(chain_input, callbacks=[cb])

            answer = res.get("answer", "Sorry, I couldn't process that.")
            full_response = answer

            source_documents = res.get("source_documents", [])
            text_elements = []
            if source_documents:
                for i, source_doc in enumerate(source_documents):
                    source_display_name = f"Source {i+1}"
                    text_elements.append(
                        cl.Text(
                            content=source_doc.page_content,
                            name=source_display_name,
                            display="inline",
                        )
                    )

            response_message.content = full_response

            if text_elements:
                response_message.elements = text_elements

            await response_message.update()
            logger.info(f"DEBUG: RAG response processed for session {session_id}.")

        else:
            logger.info(f"DEBUG: Using default OpenAI call for session {session_id}.")
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message_history,
                temperature=0.7,
                max_tokens=800,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    await response_message.stream_token(content)
            await response_message.update()
            logger.info(
                f"DEBUG: Default OpenAI response received for session {session_id}."
            )

        if full_response:
            message_history.append({"role": "assistant", "content": full_response})
            cl.user_session.set("message_history", message_history)
            save_session(session_id, user_id, message_history)
            logger.info(
                f"DEBUG: Saved history for session {session_id}, user {user_id}. Length: {len(message_history)}"
            )
        else:
            logger.warning(f"DEBUG: No response generated for session {session_id}.")

    except Exception as e:
        logger.exception(f"ERROR processing message for session {session_id}: {e}")
        await cl.Message(content=f"An error occurred: {e}", author="Error Bot").send()

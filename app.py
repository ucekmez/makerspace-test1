import os
import uuid
import tempfile
from typing import List
from datetime import datetime
import chainlit as cl
from openai import OpenAI
from loguru import logger

# Import custom DB utils
from db_utils import init_db, get_sessions, load_session, save_session

# Langchain / RAG specific imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma # Community package
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader # Community package
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Import message types

# Initialize the custom database
init_db()

# Initialize OpenAI client...
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.info("Warning: OPENAI_API_KEY not found.")
# Make sure OPENAI_API_KEY is set before initializing Langchain components
if not api_key:
     raise ValueError("OPENAI_API_KEY environment variable not set. Cannot initialize RAG components.")

client = OpenAI(api_key=api_key)

# Configure Chainlit settings
os.environ["CHAINLIT_AUTH_SECRET"] = "secret"
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"
os.environ["CHAINLIT_HOST"] = "0.0.0.0"
os.environ["CHAINLIT_PORT"] = "7860"

# --- RAG Configuration --- 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Updated system template to be more conversational and less strict on formatting
system_template = """You are a helpful assistant. Use the following context to answer the user's question accurately. 
If the context doesn't contain the answer, say you don't know. 
Refer to the source documents when possible.
Context:
{context}

User Question: {question}
Answer:"""

# Using Langchain Core message types for prompt template construction
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Default System message (used when no RAG is active)
DEFAULT_SYSTEM_MESSAGE = {"role": "system", "content": """
You are a highly humorous, witty, and friendly AI assistant with an endless supply of jokes, puns, and playful sarcasm. 
Your personality is lighthearted, fun, and engaging, making every conversation feel like chatting with a hilarious best friend.
Always respond short and concise in max 3 sentences.
""".strip()}

# --- Helper Function for RAG File Processing ---
def process_pdf_file(file_content: bytes, file_name: str) -> List[Document]:
    """Loads PDF content, splits it, and creates Langchain Documents."""
    docs = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        logger.info(f"DEBUG: Loading PDF from temporary file: {temp_file_path}")
        loader = PyPDFLoader(temp_file_path)
        # Load and split using the loader itself if possible, or load then split
        loaded_docs = loader.load() # PyPDFLoader loads full docs
        
        # Split the loaded documents
        split_docs = text_splitter.split_documents(loaded_docs)

        # Add source metadata (using filename and page number from loader)
        for i, doc in enumerate(split_docs):
             # Ensure metadata exists and add our source info
             doc.metadata = doc.metadata if doc.metadata else {}
             doc.metadata["source"] = f"{file_name} (chunk {i+1})" 
             # Keep page number if available from loader
             if 'page' in doc.metadata:
                  doc.metadata["source"] = f"{file_name} (page {doc.metadata['page'] + 1}, chunk {i+1})"
             docs.append(doc)

        logger.info(f"DEBUG: Processed {len(docs)} chunks from {file_name}.")

    except Exception as e:
        logger.error(f"Error processing PDF file {file_name}: {e}")
        # Optionally raise or handle error differently
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"DEBUG: Removed temporary file: {temp_file_path}")
            except OSError as oe:
                 logger.error(f"Error removing temporary file {temp_file_path}: {oe}")
    return docs

# --- Chainlit Authentication --- 
@cl.password_auth_callback
async def auth_callback(credentials) -> cl.User | None:
    logger.info(f"DEBUG: auth_callback received: {credentials}")
    if isinstance(credentials, str) and credentials == "admin":
        logger.info("DEBUG: User 'admin' authenticated.")
        return cl.User(identifier="admin") 
    else:
        logger.warning(f"DEBUG: Auth failed for '{credentials}'.")
        return None

# --- Upload Request Handler --- 
async def handle_upload_request():
    """Handles the upload request triggered by typing 'upload'."""
    app_user = cl.user_session.get("user")
    session_id = cl.user_session.get("session_id")
    if not app_user or not session_id:
        await cl.ErrorMessage("Login and active session required to upload documents.").send()
        return
    user_id = app_user.identifier

    # Ask for file
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to use for Q&A!",
        accept={"application/pdf": [".pdf"]},
        max_size_mb=50,
        timeout=300,
    ).send()

    if not files:
        await cl.Message(content="No file uploaded.").send()
        return

    file = files[0] # file is an AskFileResponse object
    
    # --- Read file content from path --- 
    file_bytes = None
    file_path = None
    try:
        # Check if path attribute exists
        if not hasattr(file, 'path') or not file.path:
             logger.error("Error: AskFileResponse object does not have a valid 'path' attribute.")
             await cl.ErrorMessage("Failed to get file path from upload.").send()
             return
        
        file_path = file.path
        logger.info(f"DEBUG: Reading uploaded file from path: {file_path}")
        # Read bytes from the temporary path provided by Chainlit
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        if not file_bytes:
            logger.error(f"Error: Read 0 bytes from file path: {file_path}")
            await cl.ErrorMessage("Failed to read content from uploaded file.").send()
            return
            
    except Exception as e:
        logger.error(f"Error reading file from path {getattr(file, 'path', 'N/A')}: {e}")
        await cl.ErrorMessage(f"Error reading uploaded file: {e}").send()
        return
    # --- End Read file content --- 

    # Notify user of processing
    proc_msg = cl.Message(content=f"Processing `{file.name}`...") 
    await proc_msg.send()

    # Process the PDF file using the read bytes
    # Pass file_bytes instead of file.content
    docs = await cl.make_async(process_pdf_file)(file_bytes, file.name)
    
    if not docs:
         await cl.ErrorMessage(f"Could not process file '{file.name}'. Please try another PDF.").send()
         await proc_msg.remove()
         return

    # Create Chroma vector store asynchronously
    logger.info(f"DEBUG: Creating vector store for session {session_id}...")
    embeddings = OpenAIEmbeddings()
    try:
        docsearch = await cl.make_async(Chroma.from_documents)(docs, embeddings)
        logger.info(f"DEBUG: Vector store created successfully for session {session_id}.")
    except Exception as e:
         logger.error(f"Error creating Chroma vector store: {e}")
         await cl.ErrorMessage("Failed to create vector store from the document.").send()
         await proc_msg.remove()
         return

    # Create ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True),
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    
    # Store the chain in the user session
    cl.user_session.set("rag_chain", chain)
    logger.info(f"DEBUG: RAG chain created and stored for session {session_id}.")

    # Let the user know the RAG system is ready
    proc_msg.content = f"Finished processing `{file.name}`. You can now ask questions about this document!"
    proc_msg.author = "RAG Setup"
    await proc_msg.update()

# --- Chainlit Lifecycle Hooks --- 
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

    # --- Custom Session Selection --- 
    existing_sessions = get_sessions(user_id)
    session_options = {}
    prompt_lines = [f"Welcome {user_id}! Choose an option:", "- Type 'new' for a new chat."]
    if existing_sessions:
        prompt_lines.append("- Type a Session ID to resume:")
        for session in existing_sessions[:5]:
            last_updated_str = datetime.fromisoformat(session['last_updated']).strftime('%Y-%m-%d %H:%M')
            sid = session['session_id']
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
        await cl.Message(content="Starting fresh! Ask general questions or type 'upload' to process a PDF.").send()
    elif user_input and user_input in session_options:
        session_id = session_options[user_input]
        message_history = load_session(session_id, user_id)
        if not message_history:
             session_id = str(uuid.uuid4())
             message_history = [DEFAULT_SYSTEM_MESSAGE]
             save_session(session_id, user_id, message_history)
             await cl.Message(content="Couldn't load session. Starting new. Type 'upload' to process a PDF.").send()
        else:
             await cl.Message(content=f"Resuming chat... Ask questions or type 'upload' to process a PDF.").send()
             # Display previous messages
             for msg in message_history:
                 if msg["role"] != "system":
                     author = user_id if msg["role"] == "user" else "Assistant"
                     await cl.Message(content=msg["content"], author=author).send()
    else: # Default to new
        session_id = str(uuid.uuid4())
        message_history = [DEFAULT_SYSTEM_MESSAGE]
        save_session(session_id, user_id, message_history)
        await cl.Message(content="Okay, starting a new chat. Ask questions or type 'upload' to process a PDF.").send()

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

    # --- Check for Upload Command --- 
    if message.content.strip().lower() == "upload":
        logger.info(f"DEBUG: Received 'upload' command from user {app_user.identifier}")
        await handle_upload_request() # Call the upload handler
        return # Stop further processing for the 'upload' message itself
    # --- End Upload Command Check --- 
    
    rag_chain = cl.user_session.get("rag_chain") 

    if not app_user or not session_id or message_history is None:
         logger.error("ERROR: Missing user/session/history in on_message!")
         await cl.ErrorMessage("Session error. Please refresh.").send()
         return
         
    user_id = app_user.identifier
    logger.info(f"DEBUG: Received message for session {session_id}, user {user_id}. RAG active: {rag_chain is not None}")

    # Add user message to our standard history list
    message_history.append({"role": "user", "content": message.content})

    try:
        response_message = cl.Message(content="")
        await response_message.send()

        full_response = ""
        text_elements = []

        if rag_chain:
            # --- Use RAG Chain --- 
            logger.info(f"DEBUG: Using RAG chain for session {session_id}.")
            cb = cl.AsyncLangchainCallbackHandler(
                stream_final_answer=True,
                answer_prefix_tokens=["Answer:"]
            )
            
            # --- Convert history to Langchain Message Objects --- 
            langchain_history = []
            # Iterate through our history, starting *after* the system message if present
            history_to_convert = message_history
            if history_to_convert and history_to_convert[0]["role"] == "system":
                 history_to_convert = history_to_convert[1:] # Skip system message for chain history
                 
            for msg_dict in history_to_convert:
                 role = msg_dict.get("role")
                 content = msg_dict.get("content", "")
                 if role == "user":
                     langchain_history.append(HumanMessage(content=content))
                 elif role == "assistant":
                     langchain_history.append(AIMessage(content=content))
            # --- End History Conversion --- 

            # Prepare input for the RAG chain with converted history
            chain_input = {
                "question": message.content,
                "chat_history": langchain_history 
            }

            # Call the RAG chain - Streaming handled by callback
            res = await rag_chain.acall(chain_input, callbacks=[cb])
            
            # Get the final answer (streaming handled by cb)
            answer = res.get("answer", "Sorry, I couldn't process that.")
            full_response = answer 
            
            # Process source documents to create inline elements
            source_documents = res.get("source_documents", [])
            text_elements = [] # Reset elements list
            if source_documents:
                for i, source_doc in enumerate(source_documents):
                    source_display_name = f"Source {i+1}"
                    text_elements.append(
                        cl.Text(
                            content=source_doc.page_content, 
                            name=source_display_name, 
                            display="inline"
                        )
                    )
            
            # Set the final content to JUST the answer
            response_message.content = full_response
            
            # Attach the inline source elements if they exist
            if text_elements:
                response_message.elements = text_elements
            
            # Update the message with final answer content and attached elements
            await response_message.update()
            logger.info(f"DEBUG: RAG response processed for session {session_id}.")

        else:
            # --- Use Default OpenAI Call --- 
            logger.info(f"DEBUG: Using default OpenAI call for session {session_id}.")
            stream = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=message_history, 
                temperature=0.7,
                max_tokens=800, 
                stream=True
            )
            for chunk in stream: 
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    await response_message.stream_token(content)
            await response_message.update()
            logger.info(f"DEBUG: Default OpenAI response received for session {session_id}.")

        # --- Common Logic: Update History and Save --- 
        if full_response:
            # Add assistant response to history list
            message_history.append({"role": "assistant", "content": full_response})
            # Update message history in the user session state
            cl.user_session.set("message_history", message_history)
            # Save the updated history to the custom database
            save_session(session_id, user_id, message_history)
            logger.info(f"DEBUG: Saved history for session {session_id}, user {user_id}. Length: {len(message_history)}")
        else:
             logger.warning(f"DEBUG: No response generated for session {session_id}.")

    except Exception as e:
        logger.exception(f"ERROR processing message for session {session_id}: {e}")
        await cl.Message(content=f"An error occurred: {e}", author="Error Bot").send() 
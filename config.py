import os
import uuid
import tempfile
from typing import List
from datetime import datetime
import chainlit as cl
from openai import OpenAI
from loguru import logger
from db_utils import init_db, get_sessions, load_session, save_session
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Community package
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader  # Community package
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

init_db()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.info("Warning: OPENAI_API_KEY not found.")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. Cannot initialize RAG components."
    )

client = OpenAI(api_key=api_key)


os.environ["CHAINLIT_AUTH_SECRET"] = "secret"
os.environ["CHAINLIT_ALLOW_ORIGINS"] = "*"
os.environ["CHAINLIT_HOST"] = "0.0.0.0"
os.environ["CHAINLIT_PORT"] = "7860"


def process_pdf_file(file_content: bytes, file_name: str) -> List[Document]:
    """Loads PDF content, splits it, and creates Langchain Documents."""
    docs = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        logger.info(f"DEBUG: Loading PDF from temporary file: {temp_file_path}")
        loader = PyPDFLoader(temp_file_path)
        loaded_docs = loader.load()

        split_docs = text_splitter.split_documents(loaded_docs)

        for i, doc in enumerate(split_docs):
            doc.metadata = doc.metadata if doc.metadata else {}
            doc.metadata["source"] = f"{file_name} (chunk {i+1})"
            if "page" in doc.metadata:
                doc.metadata[
                    "source"
                ] = f"{file_name} (page {doc.metadata['page'] + 1}, chunk {i+1})"
            docs.append(doc)

        logger.info(f"DEBUG: Processed {len(docs)} chunks from {file_name}.")

    except Exception as e:
        logger.error(f"Error processing PDF file {file_name}: {e}")
    finally:
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"DEBUG: Removed temporary file: {temp_file_path}")
            except OSError as oe:
                logger.error(f"Error removing temporary file {temp_file_path}: {oe}")
    return docs



async def handle_upload_request():
    """Handles the upload request triggered by typing 'upload'."""
    app_user = cl.user_session.get("user")
    session_id = cl.user_session.get("session_id")
    if not app_user or not session_id:
        await cl.ErrorMessage(
            "Login and active session required to upload documents."
        ).send()
        return
    user_id = app_user.identifier

    files = await cl.AskFileMessage(
        content="Please upload a PDF file to use for Q&A!",
        accept={"application/pdf": [".pdf"]},
        max_size_mb=50,
        timeout=300,
    ).send()

    if not files:
        await cl.Message(content="No file uploaded.").send()
        return

    file = files[0]  # file is an AskFileResponse object
    file_bytes = None
    file_path = None
    try:
        if not hasattr(file, "path") or not file.path:
            logger.error(
                "Error: AskFileResponse object does not have a valid 'path' attribute."
            )
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
        logger.error(
            f"Error reading file from path {getattr(file, 'path', 'N/A')}: {e}"
        )
        await cl.ErrorMessage(f"Error reading uploaded file: {e}").send()
        return

    proc_msg = cl.Message(content=f"Processing `{file.name}`...")
    await proc_msg.send()

    docs = await cl.make_async(process_pdf_file)(file_bytes, file.name)

    if not docs:
        await cl.ErrorMessage(
            f"Could not process file '{file.name}'. Please try another PDF."
        ).send()
        await proc_msg.remove()
        return

    logger.info(f"DEBUG: Creating vector store for session {session_id}...")
    embeddings = OpenAIEmbeddings()
    try:
        docsearch = await cl.make_async(Chroma.from_documents)(docs, embeddings)
        logger.info(
            f"DEBUG: Vector store created successfully for session {session_id}."
        )
    except Exception as e:
        logger.error(f"Error creating Chroma vector store: {e}")
        await cl.ErrorMessage("Failed to create vector store from the document.").send()
        await proc_msg.remove()
        return

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True),
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    cl.user_session.set("rag_chain", chain)
    logger.info(f"DEBUG: RAG chain created and stored for session {session_id}.")

    proc_msg.content = f"Finished processing `{file.name}`. You can now ask questions about this document!"
    proc_msg.author = "RAG Setup"
    await proc_msg.update()

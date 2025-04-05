import sqlite3
import json
from datetime import datetime
from loguru import logger
import os

DB_FILE = "chat_history.db"

def init_db():
    """Initializes the database and creates the sessions table if it doesn't exist."""
    logger.info(f"DEBUG: Initializing custom database at {os.path.abspath(DB_FILE)}...")
    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Corrected SQL: Removed Python comment
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_identifier TEXT NOT NULL DEFAULT 'default_user', 
                history TEXT
            )
        """)
        conn.commit() # Commit after CREATE TABLE IF NOT EXISTS
        logger.info("DEBUG: sessions table checked/created.")
        
        # Attempt to add user_identifier column (for backward compatibility)
        try:
            # Use ALTER TABLE ... ADD COLUMN syntax, ensures idempotency
            cursor.execute("ALTER TABLE sessions ADD COLUMN user_identifier TEXT NOT NULL DEFAULT 'default_user'")
            conn.commit() # Commit after successful ALTER TABLE
            logger.info("DEBUG: Added 'user_identifier' column to sessions table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                logger.info("DEBUG: 'user_identifier' column already exists.")
                pass # Column already exists, ignore
            else:
                logger.error(f"Error altering table: {e}")
                raise # Re-raise other errors
        
        logger.info("DEBUG: Custom database initialized successfully.")
        
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("DEBUG: Database connection closed.")

def get_sessions(user_identifier: str):
    """Retrieves sessions for a specific user, ordered by last update."""
    logger.info(f"DEBUG: Fetching existing sessions for user: {user_identifier}...")
    sessions = []
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, created_at, last_updated FROM sessions WHERE user_identifier = ? ORDER BY last_updated DESC", (user_identifier,))
        sessions = [dict(row) for row in cursor.fetchall()]
        logger.info(f"DEBUG: Found {len(sessions)} sessions for user {user_identifier}.")
    except sqlite3.Error as e:
        logger.error(f"Error fetching sessions for user {user_identifier}: {e}")
    finally:
        if conn:
            conn.close()
    return sessions

def load_session(session_id: str, user_identifier: str):
    """Loads the message history for a specific session ID, ensuring it belongs to the user."""
    logger.info(f"DEBUG: Loading session: {session_id} for user: {user_identifier}")
    history = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Verify user ownership before loading
        cursor.execute("SELECT history FROM sessions WHERE session_id = ? AND user_identifier = ?", (session_id, user_identifier))
        result = cursor.fetchone()
        if result and result[0]:
            history = json.loads(result[0])
            logger.info(f"DEBUG: Session {session_id} loaded successfully for user {user_identifier}, history length: {len(history)}")
        elif result is None:
             logger.warning(f"DEBUG: Session {session_id} not found or does not belong to user {user_identifier}.")
        else: # result[0] is None or empty
            logger.warning(f"DEBUG: No history found or empty for session: {session_id}")
            # Return default initial history if nothing found
            history = [{"role": "system", "content": "You are a helpful AI assistant."}]
    except sqlite3.Error as e:
        logger.error(f"Error loading session {session_id}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON history for session {session_id}: {e}")
        history = [{"role": "system", "content": "You are a helpful AI assistant."}] # Fallback
    finally:
        if conn:
            conn.close()
    return history

def save_session(session_id: str, user_identifier: str, history: list):
    """Saves or updates the message history for a session ID and user."""
    logger.info(f"DEBUG: Saving session: {session_id} for user: {user_identifier}, history length: {len(history)}")
    now = datetime.now()
    history_json = json.dumps(history)
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Use INSERT OR REPLACE (UPSERT) to handle creation and update for the specific user
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, user_identifier, created_at, last_updated, history)
            VALUES (?, ?, COALESCE((SELECT created_at FROM sessions WHERE session_id = ?), ?), ?, ?)
        """, (session_id, user_identifier, session_id, now, now, history_json))
        conn.commit()
        logger.info(f"DEBUG: Session {session_id} saved successfully for user {user_identifier}.")
    except sqlite3.Error as e:
        logger.error(f"Error saving session {session_id} for user {user_identifier}: {e}")
    finally:
        if conn:
            conn.close()

# Initialize the database when the module is loaded
init_db() 
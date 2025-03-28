---
title: AI Makerspace Test Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# GPT-4o-mini Chainlit Chatbot

A simple chatbot application built with Chainlit and OpenAI's GPT-4o-mini model.

## Features

- Interactive chat interface powered by Chainlit
- Integration with OpenAI's GPT-4o-mini model
- Persistent conversation history
- Streaming responses for better user experience

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file and add your actual OpenAI API key.

## Usage

1. Run the application:
   ```
   chainlit run app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Start chatting with the AI assistant!

## Docker Deployment

You can run the application using Docker:

1. Build the Docker image:
   ```
   docker build -t gpt4o-mini-chatbot .
   ```

2. Run the Docker container:
   ```
   docker run -p 7860:7860 -e OPENAI_API_KEY=your_api_key_here gpt4o-mini-chatbot
   ```

3. Access the application at http://localhost:7860

## Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "New Space"
   - Choose a name for your Space
   - Select "Docker" as the SDK
   - Make the Space "Public" or "Private"

2. Push your code to the Space repository:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```

3. Set your OpenAI API key as a secret in the Space:
   - Go to your Space's settings
   - Navigate to the "Variables and Secrets" tab
   - Add a new secret with name `OPENAI_API_KEY` and your API key as the value

4. The Space will automatically build and deploy your Docker container

## Project Structure

- `app.py`: Main application code
- `chainlit.md`: Welcome screen configuration
- `requirements.txt`: Required Python packages
- `.env.example`: Template for environment variables
- `Dockerfile`: Docker configuration for deployment
- `.huggingface-space`: Configuration for Hugging Face Spaces

## Notes

- The conversation history is maintained within the session
- The application uses the gpt-4o-mini model for efficient responses
- Adjust the model parameters in app.py as needed for different response styles 
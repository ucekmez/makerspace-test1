FROM python:3.10-slim

WORKDIR /app

RUN chmod -R 777 /app/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY chainlit.md .
COPY .env.example .

# Create a .env file (will need to be populated with API key at runtime)
RUN cp .env.example .env

# Set environment variables for Chainlit
ENV CHAINLIT_AUTH_SECRET=secret
ENV CHAINLIT_ALLOW_ORIGINS=*
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=7860
ENV CHAINLIT_SERVER_PORT=7860

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Command to run the application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"] 
# Dockerfile for LangChain Agent API
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

# Copy the application code
COPY . .

# Create a .env file if it doesn't exist
RUN touch .env

# Expose the FastAPI port
EXPOSE 5005

# Set environment variables (these can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=5005

# These will be overridden by .env file or at runtime
ENV API_KEY=default-api-key
ENV OPENAI_API_KEY=""
ENV OPENAI_BASE_URL=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5005/health || exit 1

# Command to run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5005"]
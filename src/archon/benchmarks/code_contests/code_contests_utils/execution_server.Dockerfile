# from this directory:
# docker build -t code-contests-python-execution-server -f execution_server.Dockerfile .

# docker run -p 8004:8004 code-contests-python-execution-server

# Use an official Python runtime as the parent image
FROM python:3.11-slim
RUN apt update && apt-get install -y curl tmux && apt-get clean

# Install the required dependencies
RUN pip install fastapi uvicorn typer

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY code_contests/code_contests_utils/execution_server.py /app
COPY code_contests/code_contests_utils/schema.py /app
COPY code_contests/code_contests_utils/compare_results.py /app

# Make port 8005 available to the world outside this container
EXPOSE 8005

# Run the FastAPI server when the container launches

CMD ["python", "execution_server.py"]
# Use the official Python image from DockerHub
FROM python:3.9-slim

# Set a directory for the app
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run on container start
CMD ["streamlit", "run", "fcg-app.py", "--theme.base", "dark"]
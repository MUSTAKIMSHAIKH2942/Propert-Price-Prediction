# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /flask_app

# Copy requirements.txt and install dependencies
COPY docker_shared/requirements.txt /flask_app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY docker_shared/ /flask_app/

# Expose port 8080 to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]

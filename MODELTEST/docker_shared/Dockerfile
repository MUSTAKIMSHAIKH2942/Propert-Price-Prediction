# Use the official Python image from the Docker Hub
FROM python:3.13

# Set the working directory
WORKDIR /flask_app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY docker_shared/ /flask_app/

# Expose port 8080 to the world outside this container
EXPOSE 8080

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

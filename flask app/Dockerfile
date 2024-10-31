# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /Flask_APP

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY app/ ./

# Copy the DATA directory into the container
COPY DATA/ ./DATA

# Copy the credentials file into the container
COPY credentials.json ./

# Set the environment variable for Google credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/Flask_APP/credentials.json"

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run main.py when the container launches
CMD ["python", "main.py"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages
RUN pip install numpy pandas tensorflow flask matplotlib

# Expose port 5000 for the app
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
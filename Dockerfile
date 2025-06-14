# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any OS-level dependencies required by your Python packages
# For example, if you encounter issues with building numpy/pandas, you might need build-essential or libgfortran
# For most cases, the slim-buster image should be sufficient with pre-compiled wheels.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port your application will run on
EXPOSE 8000

# Run the start script
# CMD ["./start.sh"] # We will use render.yaml to specify the command

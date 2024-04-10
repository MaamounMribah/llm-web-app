# Use an official PyTorch runtime as a parent image
FROM python:3.10   

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install  flask transformers torch  google-cloud-storage --progress-bar off
ENV PYTORCH_CUDA_ALLOC_CONF 'expandable_segments:True'

# Make port 5000 available to the world outside this container
EXPOSE 5000



# Run app.py when the container launches
ENTRYPOINT ["python3", "app.py"]

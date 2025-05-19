# Use the official TensorFlow Docker image as the base image  
FROM tensorflow/tensorflow:2.10.0  
  
# Set the working directory inside the container  
WORKDIR /app  
  
# Copy the requirements.txt file into the working directory  
COPY requirements.txt .  
  
# Install the required Python packages  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Copy the rest of the application files into the working directory  
COPY . .  
  
# Define the command to run the Python script  
CMD ["python", "SARSA.py"]  

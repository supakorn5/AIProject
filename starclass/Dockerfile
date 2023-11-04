FROM python:3.11.4

# Set the working directory
WORKDIR /projectAI

# Install system-level dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Copy requirements and install Python packages
COPY ./requirements.txt /projectAI/starclass/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /projectAI/starclass/requirements.txt

RUN pip install scikit-learn
# Copy your application code
COPY ./app /projectAI/app
COPY ./model /projectAI/starclass/model

# Copy the 'genstar.pkl' file into the container

# Set the Python path
ENV PYTHONPATH "${PYTHONPATH}:/projectAI"

# Start your application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

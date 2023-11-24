# Use Python 3.10 as the base image with minimal OS specifics
FROM python:3.10

# Install Poetry for Python dependency management
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && . $HOME/.poetry/env \
    && poetry --version

# Set the working directory inside the container to /app
WORKDIR /app

# Copy following two files to the container's /app directory
# COPY pyproject.toml poetry.lock /app

# Copy everything from the local directory to the /app directory in the container
COPY . /app

# Configure Poetry not to create virtual enviroment and to install dependencies in pyproject.toml
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev

# Specify network port to use during runtime. Default port for Streamlit is 8501
EXPOSE 8501

# Commands to run when the container starts
CMD ["streamlit", "run", "your_app_file.py"]

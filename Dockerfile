FROM python:3.10-buster as builder

ENV POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install poetry==$POETRY_VERSION

WORKDIR /app

COPY pyproject.toml poetry.lock README.md /app

RUN poetry install --without dev && poetry cache clear --all .



FROM python:3.10-slim-buster as runtime

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY main /app/main
#COPY pyproject.toml poetry.lock README.md /app
#RUN pip install poetry && poetry install && pip uninstall -y poetry

EXPOSE 8501

CMD ["streamlit", "run", "/app/main/Main.py"]

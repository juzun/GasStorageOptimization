FROM python:3.10-buster as builder

ENV POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install poetry==$POETRY_VERSION

WORKDIR /app

COPY pyproject.toml poetry.lock README.md /app/

RUN poetry install --without dev && poetry cache clear --all .


FROM python:3.10-slim-bullseye as runtime

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libopenblas-dev libtbb2 && \
    apt-get clean

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PATH="/app/src/solver/SCIPOptSuite-8.0.4-Linux/bin:${PATH}"

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY src /app/src

EXPOSE 8501

CMD ["streamlit", "run", "/app/src/Main.py"]

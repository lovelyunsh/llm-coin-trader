FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction --no-ansi

COPY coin_trader/ coin_trader/
RUN poetry install --only main --no-interaction --no-ansi

EXPOSE 8932

CMD ["python", "-m", "coin_trader.main", "web", "--host", "0.0.0.0", "--port", "8932"]

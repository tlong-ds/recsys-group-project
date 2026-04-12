FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt README.md pyproject.toml /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY configs /app/configs
COPY models /app/models
COPY streamlit_app /app/streamlit_app

EXPOSE 8000

CMD ["recsys-serve", "--config", "configs/serving_config.yaml"]

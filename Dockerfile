FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt README.md pyproject.toml /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

COPY configs /app/configs
COPY models /app/models
COPY streamlit_app /app/streamlit_app

RUN addgroup --system --gid 10001 recsys \
    && adduser --system --uid 10001 --ingroup recsys recsys \
    && chown -R recsys:recsys /app

USER recsys

EXPOSE 8000

CMD ["recsys-serve", "--config", "configs/serving_config.yaml"]

ARG PYTHON_BASE_IMAGE=python:3.12.10-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

ARG TESSDATA_BEST_REVISION=e12c65a915945e4c28e237a9b52bc4a8f39a0cec
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROFILE_INSIGHT_TESSDATA_DIR=/usr/local/share/tessdata_best \
    HOME=/tmp

RUN set -eux; \
    apt-get update; \
    apt-get install --yes --no-install-recommends ca-certificates curl libgomp1 tesseract-ocr; \
    mkdir -p "${PROFILE_INSIGHT_TESSDATA_DIR}"; \
    for language in fas eng osd; do \
      curl --fail --location --silent --show-error --retry 3 \
        "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/${TESSDATA_BEST_REVISION}/${language}.traineddata" \
        --output "${PROFILE_INSIGHT_TESSDATA_DIR}/${language}.traineddata"; \
      test -s "${PROFILE_INSIGHT_TESSDATA_DIR}/${language}.traineddata"; \
    done; \
    tesseract --tessdata-dir "${PROFILE_INSIGHT_TESSDATA_DIR}" --list-langs | grep --fixed-strings --line-regexp --quiet fas; \
    tesseract --tessdata-dir "${PROFILE_INSIGHT_TESSDATA_DIR}" --list-langs | grep --fixed-strings --line-regexp --quiet eng; \
    tesseract --tessdata-dir "${PROFILE_INSIGHT_TESSDATA_DIR}" --list-langs | grep --fixed-strings --line-regexp --quiet osd; \
    apt-get purge --yes --auto-remove curl; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/

RUN set -eux; \
    groupadd --gid 10001 app; \
    useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app; \
    chmod 0755 /usr/local/share/tessdata_best; \
    chmod 0644 /usr/local/share/tessdata_best/*.traineddata; \
    chown -R 10001:10001 /home/app /app

USER 10001:10001
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.host.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

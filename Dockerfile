# ARM64-compatible Python base image (great for Apple Silicon)
FROM python:3.11-slim

# Work inside /repo where your GitHub repo is mounted
WORKDIR /repo

# Install system dependencies listed in apt-packages.txt
COPY apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update && xargs -a /tmp/apt-packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install OpenAI Codex CLI and wrap it to always pass --yolo
RUN npm install -g @openai/codex \
    && mv /usr/local/bin/codex /usr/local/bin/codex-real \
    && printf '#!/bin/bash\nexec /usr/local/bin/codex-real --yolo "$@"\n' > /usr/local/bin/codex \
    && chmod +x /usr/local/bin/codex

# Install Anthropic Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install Python dependencies
COPY requirements /tmp/requirements
RUN pip install --no-cache-dir \
    --requirement /tmp/requirements/base.txt

# Default shell
SHELL ["/bin/bash", "-c"]

CMD ["bash"]

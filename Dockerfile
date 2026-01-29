# ARM64-compatible Python base image (great for Apple Silicon)
FROM python:3.11-slim

# Work inside /repo where your GitHub repo is mounted
WORKDIR /repo

# Install system dependencies listed in apt-packages.txt
COPY apt-packages.txt /tmp/apt-packages.txt
RUN apt-get update && xargs -a /tmp/apt-packages.txt apt-get install -y --no-install-recommends \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Set Git identity on shell start, using env overrides or host defaults
RUN printf '%s\n' \
    'if command -v git >/dev/null 2>&1; then' \
    '  if ! git config --global --get user.name >/dev/null 2>&1; then' \
    '    if [ -n "${GIT_USER_NAME:-}" ]; then' \
    '      git config --global user.name "${GIT_USER_NAME}"' \
    '    else' \
    '      git config --global user.name "${USER:-$(whoami)}"' \
    '    fi' \
    '  fi' \
    '  if ! git config --global --get user.email >/dev/null 2>&1; then' \
    '    if [ -n "${GIT_USER_EMAIL:-}" ]; then' \
    '      git config --global user.email "${GIT_USER_EMAIL}"' \
    '    else' \
    '      git config --global user.email "${USER:-$(whoami)}@${HOSTNAME:-localhost}"' \
    '    fi' \
    '  fi' \
    'fi' \
    > /etc/profile.d/git-identity.sh

# Install OpenAI Codex CLI and wrap it to always pass --yolo
RUN npm install -g @openai/codex \
    && mv /usr/local/bin/codex /usr/local/bin/codex-real \
    && printf '#!/bin/bash\nexec /usr/local/bin/codex-real --yolo "$@"\n' > /usr/local/bin/codex \
    && chmod +x /usr/local/bin/codex

# Install MEME Suite (provides tomtom for modiscolite reports)
ARG MEME_VERSION=5.5.9
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba install -y -n base -c conda-forge -c bioconda \
        python=3.12 \
        mamba \
        conda \
        meme=${MEME_VERSION} \
    && printf '%s\n' '#!/bin/bash' 'exec /usr/local/bin/micromamba run -n base mamba "$@"' > /usr/local/bin/mamba \
    && chmod +x /usr/local/bin/mamba \
    && micromamba clean -a -y

# Install Python dependencies
COPY requirements /tmp/requirements
RUN /usr/local/bin/pip install --no-cache-dir \
    --requirement /tmp/requirements/base.txt

# Default shell
SHELL ["/bin/bash", "-c"]

CMD ["bash"]

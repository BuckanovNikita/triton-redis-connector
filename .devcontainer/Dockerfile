FROM debian:12

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        htop \
        iproute2 \
        iputils-ping \
        net-tools \
        procps \
        sudo \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER vscode

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/vscode/.local/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN uv python install 3.12 \
    && mkdir -p /home/vscode/.claude \
    && curl -fsSL https://claude.ai/install.sh | sh

COPY .devcontainer/managed-settings.json /tmp/managed-settings.json
USER root
RUN mkdir -p /etc/claude-code \
    && mv /tmp/managed-settings.json /etc/claude-code/managed-settings.json \
    && chmod 644 /etc/claude-code/managed-settings.json
USER vscode

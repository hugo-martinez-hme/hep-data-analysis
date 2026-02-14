FROM quay.io/almalinux/almalinux:9 as builder

RUN dnf install -y python3.11 python3.11-devel gcc-c++ git && \
    dnf clean all

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalación PyTorch para CPU usando extra-index para no perder el acceso a PyPi
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    uproot \
    awkward \
    vector \
    numpy \
    pandas \
    matplotlib \
    langchain \
    pydantic \
    pytest \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

FROM quay.io/almalinux/almalinux:9-minimal

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN microdnf install -y python3.11 && \
    microdnf clean all && \
    rm -rf /var/cache/yum /var/cache/dnf

WORKDIR /app
CMD ["/bin/bash"]

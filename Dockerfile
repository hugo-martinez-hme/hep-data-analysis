FROM quay.io/almalinux/almalinux:9 as builder

RUN dnf install -y python3.11 python3.11-devel gcc-c++ git && \
    dnf clean all

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    uproot awkward vector numpy pandas matplotlib \
    langchain pydantic pytest mcp \
    pysqlite3-binary \
    chromadb \
    sentence-transformers \
    mplhep \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

FROM quay.io/almalinux/almalinux:9-minimal

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN microdnf install -y python3.11 git && \
    microdnf clean all && \
    rm -rf /var/cache/yum /var/cache/dnf

ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/pysqlite3:$LD_LIBRARY_PATH"

WORKDIR /app
CMD ["/bin/bash"]

uv sync

rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*

uv remove torchao xformers tensorflow tensorflow-cpu 2>/dev/null || true

rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*

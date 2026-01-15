## Triton learning note

### 1. Launch docker image
```bash
docker run -it --ipc=host --network=host --privileged --security-opt seccomp=unconfined --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/dri --device=/dev/mem -v /raid/users/xisun/:/root/workspace/ -v /raid/models:/models  --name xisun_triton rocm/pytorch:rocm7.1_ubuntu22.04_py3.11_pytorch_release_2.9.1
```

### 2. Check torch and triton version
```bash
pip show torch | grep Version
```
```
Version: 2.9.1+rocm7.1.0.lw.git351ff442
```
Commit: https://github.com/ROCm/pytorch/commit/351ff442fbe2b9807b8a7ae6c2c30f448d56a736

```bash
pip show triton | grep Version
```
```
Version: 3.5.1+rocm7.1.0.gita272dfa8
```
Commit: https://github.com/ROCm/triton/commit/a272dfa85e20c6c167a1ec0fab48f7f9f4fd47c4

### 3. Build triton from source:

Triton source: https://github.com/triton-lang/triton/commit/d8da7c998c75c88691e0ec157fcfd92b49d7060a
```bash
pip uninstall triton
git clone https://github.com/triton-lang/triton && cd triton && git checkout d8da7c998c75c88691e0ec157fcfd92b49d7060a && cd ..
git clone https://github.com/llvm-project/llvm
./build_triton_debug.sh
```

### 4. Matmul kernel example
Dump triton kernel in `triton_cache` and MLIR full pass in `full.mlir`
```bash
cd matmul
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./triton_cache MLIR_ENABLE_DUMP=1 MLIR_DUMP_PATH=full.mlir python 03-matrix-multiplication.py
```

Parse MLIR pass into speprate pass, saved in `MLIR` folder
```python
python parse_mlir.py
```

Apply `--inline` pass with `triton-opt` on `source.mlir`
```bash
../build/triton-debug/bin/triton-opt MLIR/01-source.mlir --inline 2>&1 | tee inline.mlir
```

### 5. Reference
- https://www.lei.chat/posts/triton-compiler-development-tips
- https://github.com/dsl-learn/Triton-blog-file/tree/main
- https://github.com/triton-lang/triton?tab=readme-ov-file#tips-for-hacking

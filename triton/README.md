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

Triton source: https://github.com/triton-lang/triton/commit/0a73259f61a063eb9325048799a9ffb0ea7d1dec
```bash
pip uninstall triton -y
git clone https://github.com/triton-lang/triton && cd triton && git checkout 0a73259f61a063eb9325048799a9ffb0ea7d1dec
make dev-install-llvm
```
Check triton version
```
pip show triton | grep Version
```

```
Version: 3.6.0+git0a73259f
```

### 4. Matmul kernel example 
Dump triton kernel in `triton_cache` and MLIR full pass in `full.mlir`, amdgcn dump in `amdgcn_dump.log`
```bash
cd matmul
export TRITON_ALWAYS_COMPILE=1 
export TRITON_KERNEL_DUMP=1 
export TRITON_DUMP_DIR=./triton_cache 
export MLIR_ENABLE_DUMP=1 
export MLIR_DUMP_PATH=full.mlir 
export AMDGCN_ENABLE_DUMP=1
python 03-matrix-multiplication.py 2>&1 | tee amdgcn_dump.log
```

Parse MLIR pass into speprate pass, saved in `MLIR` folder
```python
python parse_mlir.py
```

Apply `--inline` pass with `triton-opt` on `source.mlir`
```bash
../triton/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt MLIR/01-source.mlir --inline 2>&1 | tee inline.mlir
```

### 5. Trobleshooting
### 5.1 Build Triton & MLIR debug from source
```bash
./build_triton_debug.sh
./build/triton-debug/bin/triton-opt matmul/MLIR/01-source.mlir --inline 2>&1 | tee inline.mlir
```

### 6. Reference
- https://www.lei.chat/posts/triton-compiler-development-tips
- https://github.com/dsl-learn/Triton-blog-file/tree/main
- https://github.com/triton-lang/triton?tab=readme-ov-file#tips-for-hacking

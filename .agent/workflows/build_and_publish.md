---
description: Build and Publish Krira Chunker
---

# Build and Publish Workflow

1. **Clean Project**
   ```powershell
   cargo clean
   Get-ChildItem -Path target -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -Recurse
   ```

2. **Environment Setup**
   Ensure you have Visual Studio C++ Build Tools installed.
   ```powershell
   # Check for compiler
   cl.exe
   ```

3. **Build Wheels**
   ```powershell
   maturin build --release
   ```

4. **Install Locally (for testing)**
   ```powershell
   pip install target/wheels/krira_augment-*.whl --force-reinstall
   ```

5. **Test**
   ```powershell
   pytest tests/
   ```

6. **Publish to PyPI**
   ```powershell
   maturin publish
   ```

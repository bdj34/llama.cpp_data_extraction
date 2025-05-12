# llama.cpp_data_extraction

This is a fork of the main [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp).  
I have added a few command-line parameters and an example called `data-extraction` for the purpose of structuring pathology reports using LLMs.

This is the code we used for the work in our preprint:  
📄 https://www.medrxiv.org/content/10.1101/2024.11.27.24318083v2

Create an issue on this repo or reach out to me at brian.d.johnson97@gmail.com or bdj001@ucsd.edu if you have questions!

---

**Supported/recommended models:**  
This fork is up to date with the main `llama.cpp` GitHub as of **Nov 27, 2024** and exists primarily for exact reproduction of our work.  
Any models released after that date will likely not work.

👉 For ongoing work and support for newer models, see my alternative fork:  
https://github.com/bdj34/llama.cpp_dev

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Demo: Air-gapped Server Setup

We focus on compiling this software on a remote server **without internet access**.

**Step 1: Clone, compress, and transfer**
```bash
git clone https://github.com/bdj34/llama.cpp_data_extraction
tar -czvf DESIRED_PATH/llama.cpp_data_extraction.tar.gz -C PATH_USED_FOR_GIT_CLONE/llama.cpp_data_extraction .
```

Transfer `llama.cpp_data_extraction.tar.gz` to the server.  
*(Example: at the VA, I transfer it via MS Teams to my VA computer, then upload it using the VINCI upload tool.)*

**Step 2: Download model**  
Visit [HuggingFace](https://huggingface.co/briandj97/models_used/tree/main) to download one of the GGUF models used in our work (or use your own).
Similarly, transfer the gguf file to the remote server
*(Example: at the VA, I split the gguf, transfer it via MS Teams to my VA computer, then email VINCI asking them to upload a large file.)*
---

## Compiling on Linux

### No GPU
```bash
mkdir llama.cpp_data_extraction
tar -xzvf llama.cpp_data_extraction.tar.gz -C llama.cpp_data_extraction/
cd llama.cpp_data_extraction
cmake -B build --fresh
cmake --build build --config Release
```

### With GPU (CUDA)
```bash
mkdir llama.cpp_data_extraction
tar -xzvf llama.cpp_data_extraction.tar.gz -C llama.cpp_data_extraction/
cd llama.cpp__data_extraction
cmake -B build -DGGML_CUDA=ON --fresh
cmake --build build --config Release
```

---

## Compiling on Windows

Unzip and expand the .tar.gz (using 7zip)
*(Untested by me — instructions copied from `llama.cpp`. Refer to the main repo for support.)*
- Building for Windows (x86, x64 and arm64) with MSVC or clang as compilers:
    - Install Visual Studio 2022, e.g. via the [Community Edition](https://visualstudio.microsoft.com/vs/community/). In the installer, select at least the following options (this also automatically installs the required additional tools like CMake,...):
    - Tab Workload: Desktop-development with C++
    - Tab Components (select quickly via search): C++-_CMake_ Tools for Windows, _Git_ for Windows, C++-_Clang_ Compiler for Windows, MS-Build Support for LLVM-Toolset (clang)
    - Please remember to always use a Developer Command Prompt / PowerShell for VS2022 for git, build, test
    - For Windows on ARM (arm64, WoA) build with:
    ```bash
    cmake --preset arm64-windows-llvm-release -D GGML_OPENMP=OFF
    cmake --build build-arm64-windows-llvm-release
    ```
    Building for arm64 can also be done with the MSVC compiler with the build-arm64-windows-MSVC preset, or the standard CMake build instructions. However, note that the MSVC compiler does not support inline ARM assembly code, used e.g. for the accelerated Q4_0_N_M CPU kernels.

    For building with ninja generator and clang compiler as default:
      -set path:set LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\lib\x64\uwp;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64
      ```bash
      cmake --preset x64-windows-llvm-release
      cmake --build build-x64-windows-llvm-release
      ```
---

## Compiling on macOS (Apple Silicon)

*(I have tested this and it should work. Let me know if there are issues.)*
```bash
cmake -B build
cmake --build build --config Release
```

---

## Example running command
```bash
cd DESIRED_PATH/llama.cpp_data_extraction
mkdir -p ../testing_CRC_extraction_outDir

./build/bin/data-extraction --extractionType crc \
-m ~/Downloads/models_gguf/gemma2-9B_f16.gguf \
--sequences 16 --parallel 16 --n-predict 300 \
--batch-size 2048 --n-gpu-layers 99 --ctx-size 20000 \
--temp 0 \
--promptStartingNumber 0 \
--patientFile ./example_data/fake_patientIDs.txt \
--grammar-file ./grammars/yesNo_grammar.gbnf \
--outDir ../testing_CRC_extraction_outDir \
--file ./example_data/pathMaybe.txt \
--promptFormat gemma2 
```

## Description (from main llama.cpp page)

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware — locally and in the cloud.

- Plain C/C++ implementation with zero dependencies
- First-class support for Apple Silicon (ARM NEON, Accelerate, Metal)
- AVX, AVX2, AVX512, and AMX support for x86 CPUs
- Support for 1.5–8 bit quantization
- CUDA kernels for NVIDIA GPUs; HIP support for AMD; MUSA for Moore Threads GPUs
- Vulkan and SYCL backends
- Hybrid CPU+GPU inference to enable running models larger than VRAM

Since its [inception](https://github.com/ggerganov/llama.cpp/issues/33#issuecomment-1465108022), the project has grown rapidly thanks to community contributions.  
It serves as the main playground for development of the [ggml](https://github.com/ggerganov/ggml) library.

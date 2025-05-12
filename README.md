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
*(Example: at the VA, I transfer it via MS Teams, then upload it using the VINCI upload tool.)*

**Step 2: Download model**  
Visit [HuggingFace](https://huggingface.co/briandj97/models_used/tree/main) to download one of the GGUF models used in our work (or use your own).

---

## Compiling on Linux

### No GPU
```bash
mkdir llama.cpp_data_extraction
tar -xzvf llama.cpp_IBD_hx.tar.gz -C llama.cpp_IBD_hx/
cd llama.cpp_IBD_hx
cmake -B build --fresh
cmake --build build --config Release
```

### With GPU (CUDA)
```bash
mkdir llama.cpp_data_extraction
tar -xzvf llama.cpp_IBD_hx.tar.gz -C llama.cpp_IBD_hx/
cd llama.cpp_IBD_hx
cmake -B build -DGGML_CUDA=ON --fresh
cmake --build build --config Release
```

---

## Compiling on Windows

*(Untested by me — instructions copied from `llama.cpp`. Refer to the main repo for support.)*

---

## Compiling on macOS (Apple Silicon)

*(TBD — add instructions if/when tested.)*

---

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

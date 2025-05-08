# llama.cpp_data_extraction

This is a fork of the main llama.cpp github found at: https://github.com/ggml-org/llama.cpp 
I have added a few command-line parameters and an example called data-extraction for the purpose of structuring pathology reports using LLMs.
This is the code we used for the work in our preprint: https://www.medrxiv.org/content/10.1101/2024.11.27.24318083v2
Create an issue on this repo or reach out to me at brian.d.johnson97@gmail.com (bdj001@ucsd.edu) if you have questions!

**Supported/recommended models:**
This fork is up to date with the main llama.cpp github as of Nov 27, 2024 and mainly exists for exact reproduction of our work. Any models released since Nov 27, 2024 will not work. 
See https://github.com/bdj34/llama.cpp_dev for an alternative fork I created for ongoing work that has support for more recent models and should be easier to use in general.  

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

### Demo
I will focus on how to compile this software on a remote server without internet access. The first steps for this are to download this code, compress it, and transfer it to the server:
git clone https://github.com/bdj34/llama.cpp_data_extraction
tar -czvf DESIRED_PATH/llama.cpp_data_extraction.tar.gz -C PATH_USED_FOR_GIT_CLONE/llama.cpp_data_extraction .
Upload/transfer the llama.cpp_data_extraction.tar.gz file to the server somehow. As an example for the VA, I would transfer this to my VA workspace via MS Teams, then use the VINCI upload tool to upload it. This will depend on your exact configuration.


See https://huggingface.co/briandj97/models_used/tree/main to download one of the gguf models we used in our work (or use your desired model).

## Compiling on linux
I used a linux development server. Initially, we didn't have a GPU

# No GPU

# GPU
mkdir llama.cpp_data_extraction
tar -xzvf llama.cpp_IBD_hx.tar.gz -C llama.cpp_IBD_hx/
cd llama.cpp_IBD_hx
cmake -B build -DGGML_CUDA=ON --fresh
cmake --build build --config Release

## Compiling on windows (untested by me, copied from llama.cpp, see llama.cpp for support)

## Compiling on mac (apple M1 or later chips)


## Description (from main llama.cpp page)

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
variety of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads MTT GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

Since its [inception](https://github.com/ggerganov/llama.cpp/issues/33#issuecomment-1465108022), the project has
improved significantly thanks to many contributions. It is the main playground for developing new features for the
[ggml](https://github.com/ggerganov/ggml) library.


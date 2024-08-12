# Install script for directory: /Users/brianjohnson/VA_IBD/llama.cpp/tests

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-tokenizer-0")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-tokenizer-1-bpe")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-tokenizer-1-spm")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-quantize-fns")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-quantize-perf")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-sampling")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-chat-template")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-grammar-parser")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-llama-grammar")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-grammar-integration")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-grad0")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grad0" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grad0")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grad0")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grad0")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-backend-ops")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-rope")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-model-load-cancel")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-autorelease")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/brianjohnson/VA_IBD/llama.cpp/bin/test-json-schema-to-grammar")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/src"
      -delete_rpath "/Users/brianjohnson/VA_IBD/llama.cpp/ggml/src"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar")
    endif()
  endif()
endif()


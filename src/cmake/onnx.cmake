if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-win-x64-1.17.0.zip")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-osx-arm64-1.17.0.tgz")
  else()
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-osx-x86_64-1.17.0.tgz")
  endif()
else()
  message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
endif()

FetchContent_Declare(onnxruntime
  URL ${ONNX_URL}
)
FetchContent_MakeAvailable(onnxruntime)
include_directories(${onnxruntime_SOURCE_DIR}/include)
link_directories(${onnxruntime_SOURCE_DIR}/lib)

if(MSVC)
  file(GLOB ONNX_DLLS "${onnxruntime_SOURCE_DIR}/lib/*.dll")

  if(CMAKE_BUILD_TYPE)
    file(COPY ${ONNX_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
  else()
    file(COPY ${ONNX_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE_INIT})
  endif()
endif()

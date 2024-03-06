// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <sstream>
#include "diarization/onnx_model.h"
#include "glog/logging.h"

Ort::Env OnnxModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions OnnxModel::session_options_ = Ort::SessionOptions();
static char input_node_ptr_[3][32];
static char output_node_ptr_[3][32];

void OnnxModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
}

static std::wstring ToWString(const std::string& str) {
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t* p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}

OnnxModel::OnnxModel(const std::string& model_path) {
  InitEngineThreads(1);
#ifdef _MSC_VER
  session_ = std::make_shared<Ort::Session>(env_, ToWString(model_path).c_str(),
                                            session_options_);
#else
  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
#endif
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session_->GetInputCount();
  input_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    auto input_name = session_->GetInputNameAllocated(i, allocator);
    memcpy(&input_node_ptr_[i], input_name.get(), strlen(input_name.get()));
    LOG(INFO) << "Input names[" << i << "]: " << input_name.get();
    input_node_names_[i] = (const char *)&input_node_ptr_[i];
  }

  // Output info
  num_nodes = session_->GetOutputCount();
  output_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    auto output_name = session_->GetOutputNameAllocated(i, allocator);
    memcpy(&output_node_ptr_[i], output_name.get(), strlen(output_name.get()));
    LOG(INFO) << "Output names[" << i << "]: " << output_name.get();
    output_node_names_[i] = (const char *)&output_node_ptr_[i];
  }
}

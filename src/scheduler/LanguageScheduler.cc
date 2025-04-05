#include "LanguageScheduler.h"
#include "IterLevelScheduler.h"
#include <fstream>

std::unique_ptr<LangScheduler> LangScheduler::create(std::string name, std::string path, 
                                                      std::unique_ptr<LanguageModel> model,
                                                      SimulationConfig config,
                                                      json info) {
  if(info["scheduler"] == "simple") {
    spdlog::info("Simple Language scheduler selected");
    return std::make_unique<LangScheduler>(name, path, std::move(model), config, info["scheduler_config"]);
  }
  else if(info["scheduler"] == "iter") {
    spdlog::info("Iter Language scheduler selected");
    return std::make_unique<IterLevelScheduler>(name, path, std::move(model), config, info["scheduler_config"]);
  }
  else {
    spdlog::error("Invalid scheduler type: {}", info["scheduler"]);
    throw std::runtime_error("Invalid scheduler type");
    return nullptr;
  }
}

LangScheduler::LangScheduler(std::string name, std::string path, std::unique_ptr<LanguageModel> model, 
                             SimulationConfig config, json _scheduler_config) {
  _name = name;
  _config = config;
  _scheduler_config = _scheduler_config;
  _language_model = std::move(model);
  json model_config = _language_model->get_model_config();
  _run_single_layer = _language_model->is_run_single_layer();
  _num_layers = model_config["num_hidden_layers"];
  _num_layers = _num_layers / (uint64_t) model_config["pipeline_parallel_size"];
  _num_sim_layers = _run_single_layer ? 1 : _num_layers;
  _num_attention_heads = model_config["num_attention_heads"];
  _num_kv_heads = model_config["num_kv_heads"];
  _hidden_size = model_config["hidden_size"];
  // 每层的kv cache参数维度 （_num_attention_heads / _num_kv_heads）:MHA=1,GQA=G
  _cache_dim = _hidden_size / _num_attention_heads * _num_kv_heads;
  _max_seq_length = model_config["max_seq_length"];
  if(_scheduler_config.contains("max_batch_size"))
    _max_batch_size = _scheduler_config["max_batch_size"];
  else 
    _max_batch_size = 0;
  if(_scheduler_config.contains("check_mem_size"))
    _check_mem_size = _scheduler_config["check_mem_size"];
  else
    _check_mem_size = true;
  _cycle = 0;
  _max_dims = {_max_seq_length, _cache_dim};
  parse_request_trace(path); 
  spdlog::info("num layer {} num sim layer {}", _num_layers, _num_sim_layers);
}

bool LangScheduler::can_schedule_model() {
  return !_model_queue.empty();
}

std::unique_ptr<Model> LangScheduler::pop_model() {
  std::unique_ptr<Model> model = std::move(_model_queue.front());
  _model_queue.pop();
  return model;
}

void LangScheduler::cycle() {
  _cycle++;
  //Reqeust Queue to Active Requests if active is empty
  if(_active_requests.empty()) {
    while(!_request_queue.empty()) {
      if(_request_queue.front()->request_time <= _cycle) {
        init_request(_request_queue.front());
        _active_requests[_request_queue.front()->request_id] = std::move(_request_queue.front());
        _request_queue.pop();
      }
      else {
        break;
      }
      if(_max_batch_size > 0 && _active_requests.size() >= _max_batch_size) {
        break;
      }
    }
  }

  //Active Requests to Model Queue
  if(_model_queue.empty() && _requests_in_model.empty()) {
    init_inputs_and_model();
  }
}

void LangScheduler::finish_model(uint32_t model_id) {
  for(auto req_id : _requests_in_model[model_id]) {
    std::vector<uint32_t> new_cache_dim;
    // prefill阶段
    if(!_active_requests[req_id]->gen_phase) {
      uint32_t promtp_len = _active_requests[req_id]->prompt_length;
      _active_requests[req_id]->gen_phase = true;
      _active_requests[req_id]->current_length += promtp_len + 1;
    }
    // decoding阶段
    else {
      _active_requests[req_id]->current_length += 1;
    }
    new_cache_dim = {_active_requests[req_id]->current_length, _cache_dim};
    for(uint32_t i = 0; i < _num_sim_layers; i++) {
        _active_requests[req_id]->key_cache[i]->resize_tensor(new_cache_dim);
        _active_requests[req_id]->value_cache[i]->resize_tensor(new_cache_dim);
    }
    _active_requests[req_id]->running = false;
    if(_active_requests[req_id]->current_length >= _active_requests[req_id]->target_length) {
      _active_requests[req_id]->finish_time = _cycle;
      spdlog::info("Request {} completed in {} cycles", req_id, _active_requests[req_id]->finish_time - _active_requests[req_id]->start_time);
      _active_requests.erase(req_id);
    }
  }
  _requests_in_model.erase(model_id);
}

bool LangScheduler::busy() {
  return !_model_queue.empty() || !_active_requests.empty() || !_request_queue.empty();
}

// 统计kv cache占用的存储空间
uint64_t LangScheduler::get_kv_memory_size() {
  uint64_t kv_size = 0;
  for(auto iter = _active_requests.begin(); iter != _active_requests.end(); iter++) {
    for(uint32_t i = 0; i < _num_sim_layers; i++) {
      kv_size += iter->second->key_cache[i]->get_size() + iter->second->value_cache[i]->get_size();
    }
  }
  if(_run_single_layer)
    kv_size *= _num_layers;
  return kv_size;
}

void LangScheduler::parse_request_trace(std::string path) {
  std::ifstream trace_file(path);
  if (!trace_file.is_open()) {
    spdlog::error("Failed to open trace file: {}", path);
    return;
  }
  //Parse CSV input (prompt_length, target_length) and create LangRequest objects
  std::string line;
  uint32_t id = 0;
  std::getline(trace_file, line); //Skip header
  uint64_t time_offset = 0;
  while (std::getline(trace_file, line)) {
    std::stringstream ss(line);
    std::string time, prompt_length, target_length, cached_len;
    std::getline(ss, time, ',');
    std::getline(ss, prompt_length, ',');
    std::getline(ss, target_length, ',');
    std::getline(ss, cached_len, ',');
    std::unique_ptr<LangRequest> request = std::make_unique<LangRequest>();
    time_offset += std::stoull(time);
    // 根据csv中的内容创建request
    request->request_id = id++;
    request->request_time = time_offset;
    request->start_time = 0;
    request->running = false;
    request->prompt_length = std::stoi(prompt_length);
    // 最终要处理的长度：输入+输出+cached
    request->target_length = std::stoi(cached_len) + std::stoi(prompt_length) + std::stoi(target_length);
    // 当前已经cached长度
    request->current_length = std::stoi(cached_len);
    // 把创建的request内容填到request队列中，智能指针
    _request_queue.push(std::move(request));
  }
  trace_file.close();
}

// 初始化当前要处理的request
void LangScheduler::init_request(std::unique_ptr<LangRequest>& request) {
  request->start_time = _cycle;
  request->gen_phase = false;
  request->running = false;
  // resize kv cache容器容量，将其扩展到与层级一致
  request->key_cache.resize(_num_sim_layers);
  request->value_cache.resize(_num_sim_layers);
  // 维度：{当前已经存储的token长度，每层的kv cache参数维度}
  std::vector<uint32_t> first_dims = { request->current_length, _cache_dim};
  // 为每一层初始化key value的cache
  for(uint32_t i = 0; i < _num_sim_layers; i++) {
    //Allocate max_seq_length x cache_dim tensor and redefine to 0 x cache_dim
    request->key_cache[i] = 
        std::make_unique<Tensor>(_language_model->get_root_node_id(), name_gen(LAYER(i), "KeyCache"),  _max_dims, _config.precision, true);
    request->key_cache[i]->resize_tensor(first_dims);
    request->value_cache[i] = 
        std::make_unique<Tensor>(_language_model->get_root_node_id(), name_gen(LAYER(i), "ValueCache"), _max_dims, _config.precision, true);
    request->value_cache[i]->resize_tensor(first_dims);
  }
}

// 初始化输入数据，准备模型执行推理
void LangScheduler::init_inputs_and_model() {
  //Init inputs
  std::vector<LangInput> inputs;
  uint32_t num_tokens = 0;
  for(auto it = _active_requests.begin(); it != _active_requests.end(); it++) {
    if(it->second->running == false) {
      LangInput input;
      input.request_id = it->first;
      // decoding阶段
      if(it->second->gen_phase) {
        input.seq_length = 1;
        // 当前上下文长度（已经存储kv cache的token数）
        input.context_length = it->second->current_length;
      }
      // prefill阶段
      else {
        input.seq_length = it->second->prompt_length;
        // 当前上下文长度（已经存储kv cache的token数）
        input.context_length = it->second->current_length;
      }
      for(uint32_t i = 0; i < _num_sim_layers; i++) {
        // 获取当前请求kv cache智能指针管理的原始指针，将其填到input的kv cache中
        input.key_cache.push_back(it->second->key_cache[i].get());
        input.value_cache.push_back(it->second->value_cache[i].get());
      }
      // 更新统计的token长度信息
      num_tokens += input.seq_length;
      inputs.push_back(input);
    }
    if(_max_batch_size > 0 && inputs.size() >= _max_batch_size) {
      break;
    }
  }
  //Init model
  if(!inputs.empty()){
    auto infer_model = _language_model->generate_model(inputs);
    for(auto input : inputs) {
      _active_requests[input.request_id]->running = true;
      _requests_in_model[infer_model->get_id()].push_back(input.request_id);
    }
    _model_queue.push(std::move(infer_model));
    float weight_size = _language_model->get_weight_size() /(1.0 GB);
    float kv_size = get_kv_memory_size() /(1.0 GB);
    // 调用函数计算激活值需要的最大存储量:max(_qkv_out_dim, _ffn1_out_dim) * _config.precision
    float act_size = _language_model->get_act_size() /(1.0 GB) * num_tokens;
    float tot_mem = weight_size + kv_size + act_size;
    spdlog::info("Total Memory Usage: {:.2f} GB", tot_mem);
    spdlog::info("Weight Memory Usage: {:.2f} GB", weight_size);
    spdlog::info("KV Memory Usage: {:.2f} GB", kv_size);
    spdlog::info("Activation Memory Usage: {:.2f} GB", act_size);
    if(_config.dram_size < tot_mem && _config.dram_size > 0) {
      if(_check_mem_size) {
        spdlog::error("Memory Usage exceeds the memory size limit {} GB/{} GB", tot_mem, _config.dram_size);
        exit(EXIT_FAILURE);
      }
      else {
        spdlog::warn("Memory Usage exceeds the memory size limit {} GB/{} GB", tot_mem, _config.dram_size);
      }
    }
  }
}
#include "Common.h"

uint32_t generate_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}
uint32_t generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}

addr_type allocate_address(uint32_t size) {
  static addr_type base_addr{0};
  addr_type result = base_addr;
  int offset = 0;
  if (result % 256 != 0) {
    offset = 256 - (result % 256);
  }
  result += offset;
  assert(result % 256 == 0);
  base_addr += (size + offset);
  base_addr += (256 - base_addr % 256);
  return result;
}

template <typename T>
T get_config_value(json config, std::string key) {
  if (config.contains(key)) {
    return config[key];
  } else {
    throw std::runtime_error(fmt::format("Config key {} not found", key));
  }
}

const static std::map<std::string, CoreType> core_type_map = {
  {"systolic_os", CoreType::SYSTOLIC_OS},
  {"systolic_ws", CoreType::SYSTOLIC_WS}
};

const static std::map<std::string, DramType> dram_type_map = {
  {"simple", DramType::SIMPLE},
  {"ramulator", DramType::RAMULATOR1},
  {"ramulator2", DramType::RAMULATOR2}
};

const static std::map<std::string, IcntType> icnt_type_map = {
  {"simple", IcntType::SIMPLE},
  {"booksim2", IcntType::BOOKSIM2}
};

SimulationConfig initialize_config(json config) {
  SimulationConfig parsed_config;

  /* Core configs */
  // core数量
  parsed_config.num_cores = get_config_value<uint32_t>(config, "num_cores");
  // 为每个core配置一个CoreConfig结构
  parsed_config.core_config = new struct CoreConfig[parsed_config.num_cores];
  // 核心频率
  parsed_config.core_freq = get_config_value<uint32_t>(config, "core_freq");
  // 打印间隔
  parsed_config.core_print_interval = get_config_value<uint32_t>(config, "core_print_interval");

  // 单个核内的配置：脉动阵列、向量单元、各种操作延迟、SRAM
  for (int i=0; i<parsed_config.num_cores; i++) {
    std::string core_id = "core_" + std::to_string(i);
    // 在json文件中找到对应核的配置{core_config}_{core_0/1/2/3}
    auto core_config = config["core_config"][core_id];
    // core类型：SYSTOLIC_OS（部分和驻留Output Stationary）、SYSTOLIC_WS（权重驻留Weight Stationary）这里只能使用权重驻留法
    std::string core_type = core_config["core_type"];
    if (core_type_map.contains(core_type)) {
      parsed_config.core_config[i].core_type = core_type_map.at(core_type);
    } else {
      throw std::runtime_error(fmt::format("Not implemented core type {} ", core_type));
    }
    // 核的尺寸：脉动阵列中的PE数量，例子中mobile NPU为8*8
    parsed_config.core_config[i].core_width = core_config["core_width"];
    parsed_config.core_config[i].core_height = core_config["core_height"];

    /* Vector configs */
    // 向量单元处理的吞吐量 (bit)，每个core有一块vector unit
    parsed_config.core_config[i].vector_process_bit = core_config["vector_process_bit"];
    // 向量延迟设置/cycle
    parsed_config.core_config[i].add_latency = core_config["add_latency"];
    parsed_config.core_config[i].mul_latency = core_config["mul_latency"];
    parsed_config.core_config[i].exp_latency = core_config["exp_latency"];
    // 非线性操作的延迟，gelu
    parsed_config.core_config[i].gelu_latency = core_config["gelu_latency"];
  
    parsed_config.core_config[i].add_tree_latency = core_config["add_tree_latency"];
    // 标量延迟设置/cycle
    parsed_config.core_config[i].scalar_sqrt_latency = core_config["scalar_sqrt_latency"];
    parsed_config.core_config[i].scalar_add_latency = core_config["scalar_add_latency"];
    parsed_config.core_config[i].scalar_mul_latency = core_config["scalar_mul_latency"];
    parsed_config.core_config[i].mac_latency = core_config["mac_latency"];
    parsed_config.core_config[i].div_latency = core_config["div_latency"];

    /* SRAM configs */
    // SRAM的字宽
    parsed_config.core_config[i].sram_width = core_config["sram_width"];
    // Scratchpad，PE的本地高速缓存
    parsed_config.core_config[i].spad_size = core_config["spad_size"];
    // 累加器SRAM，专门存储部分和（中间计算结果）
    parsed_config.core_config[i].accum_spad_size = core_config["accum_spad_size"];
  }

  /* DRAM config */
  // DRAM类型：(ex. ramulator, simple)
  std::string dram_type = get_config_value<std::string>(config, "dram_type");
  if (dram_type_map.contains(dram_type)) {
    parsed_config.dram_type = dram_type_map.at(dram_type);
  } else {
    throw std::runtime_error(fmt::format("Not implemented dram type {} ", dram_type));
  }

  // DRAM频率
  parsed_config.dram_freq = get_config_value<uint32_t>(config, "dram_freq");
  if (config.contains("dram_latency"))
    parsed_config.dram_latency = config["dram_latency"];
  // Ramulator的DRAM配置文件
  if (config.contains("dram_config_path"))
    parsed_config.dram_config_path = config["dram_config_path"];
  // Number of DRAM channels
  parsed_config.dram_channels = config["dram_channels"];
  // DRAM request size (B)
  if (config.contains("dram_req_size"))
    parsed_config.dram_req_size = config["dram_req_size"];
  // DRAM stat print interval (cycle)
  if (config.contains("dram_print_interval"))
    parsed_config.dram_print_interval = config["dram_print_interval"];
  // Burst：一次激活传输多个数据
  // 一次burst传输占的时钟周期数量 (bust_length 8 in DDR -> 4 nbl，dram_nbl = 1)
  if(config.contains("dram_nbl"))
    parsed_config.dram_nbl = config["dram_nbl"];
  // DRAM尺寸，单位GB
  if (config.contains("dram_size"))
    parsed_config.dram_size = config["dram_size"];
  else
    parsed_config.dram_size = 0;

  /* Icnt config */
  // 互联配置类型 (ex. booksim, simple)
  std::string icnt_type = get_config_value<std::string>(config, "icnt_type");
  if (icnt_type_map.contains(icnt_type)) {
    parsed_config.icnt_type = icnt_type_map.at(icnt_type);
  } else {
    throw std::runtime_error(fmt::format("Not implemented icnt type {} ", icnt_type));
  }

  // 互联频率，单位(MHz)
  parsed_config.icnt_freq = get_config_value<uint32_t>(config, "icnt_freq");
  // 互联延迟，单位(cycle)
  if (config.contains("icnt_latency"))
    parsed_config.icnt_latency = config["icnt_latency"];
  // booksim的互连线配置文件，xx.icnt
  if (config.contains("icnt_config_path"))
    parsed_config.icnt_config_path = config["icnt_config_path"];
  if (config.contains("icnt_print_interval"))
    parsed_config.icnt_print_interval = config["icnt_print_interval"];

  // todo: 更改调度策略代码
  // 调度策略： (ex. simple, spatial_split, time_multiplex, partition_cpu)
  parsed_config.scheduler_type = get_config_value<std::string>(config, "scheduler");
  // Element's precision in tensor (Byte)，4=FP32, 2=FP16, 1=INT8 
  parsed_config.precision = get_config_value<uint32_t>(config, "precision");
  // Data Layout (NCHW,NHWC, N:Batch size, C:Channel)
  parsed_config.layout = get_config_value<std::string>(config, "layout");

  // 检查是否需要分区
  if (config.contains("partition")) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      uint32_t partition_id = uint32_t(config["partition"][core_partition]);
      parsed_config.partition_map[partition_id].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, partition_id);
    }
  } else {
    /* Default: all partition 0 */
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partition_map[0].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, 0);
    }
  }
  return parsed_config;
}

// 向上取整-除法
uint32_t ceil_div(uint32_t src, uint32_t div) { return (src+div-1)/div; }

// 输入字符串，转换成整数向量
std::vector<uint32_t> parse_dims(const std::string &str) {
  std::vector<uint32_t> dims;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, ',')) {
      dims.push_back(std::stoi(token));
  }
  return dims;
}

// 输入向量，转换成字符串
std::string dims_to_string(const std::vector<uint32_t> &dims){
  std::string str;
  for (int i=0; i<dims.size(); i++) {
    str += std::to_string(dims[i]);
    if (i != dims.size()-1) {
      str += ",";
    }
  }
  return str;
}
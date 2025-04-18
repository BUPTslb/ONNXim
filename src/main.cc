#include <fstream>
#include <chrono>
#include <filesystem>

#include "Simulator.h"
#include "helper/CommandLineParser.h"
#include "operations/OperationFactory.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the models list file");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "mode", "choose default or language mode, default = default");
  cmd_parser.add_command_line_option<std::string>(
      "trace_file", "input trace file for language mode, default = input.csv");

  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  // 获取环境变量位置
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
    std::string(onnxim_path_env) : std::string("./");
  // 模型位置
  std::string model_base_path = fs::path(onnxim_path).append("models");
  std::string level = "info";
  cmd_parser.set_if_defined("log_level", &level);
  if (level == "trace")
    spdlog::set_level(spdlog::level::trace);
  else if (level == "debug")
    spdlog::set_level(spdlog::level::debug);
  else if (level == "info")
    spdlog::set_level(spdlog::level::info);

  std::string config_path;
  cmd_parser.set_if_defined("config", &config_path);

  json config_json;
  // 打开配置文件
  std::ifstream config_file(config_path);
  if (!config_file) {
    spdlog::error("Error opening file: {}", config_path);
    exit(EXIT_FAILURE);
  }

  // 将配置文件中的内容读取到json中
  config_file >> config_json;
  config_file.close();
  // 初始化json文件
  SimulationConfig config = initialize_config(config_json);
  OperationFactory::initialize(config);

  // 运行的模型类型：（默认、language mode）
  std::string mode = "default";
  bool language_mode = false;
  cmd_parser.set_if_defined("mode", &mode);
  if (mode == "default") {
    spdlog::info("Running in default mode");
  } else if (mode == "language") {
    spdlog::info("Running in language mode");
    language_mode = true;
  } else {
    spdlog::error("Invalid mode: {}", mode);
    return 1;
  }

  // 模型列表文件，--models_list example/language_models.json
  std::string models_list_path;
  cmd_parser.set_if_defined("models_list", &models_list_path);
  std::ifstream models_list_file(models_list_path);
  if (!models_list_file) {
    spdlog::error("Error opening file: {}", models_list_path);
    exit(EXIT_FAILURE);
  }

  json models_list;
  models_list_file >> models_list;
  models_list_file.close();
  // 配置模拟器对象
  auto simulator = std::make_unique<Simulator>(config, language_mode);
  // language_models.json
  for (json model_config : models_list["models"]) {
    if(language_mode) {
      // 模型名称
      std::string model_name = model_config["name"];
      // 生成一个{model_base_path}/{language_models}/{model_name}.json字符串
      std::string model_path =
        fmt::format("{}/{}/{}.json", model_base_path, "language_models", model_name);
      // 打开 model_path 指定的文件，并将文件流对象 model_file 与该文件关联
      std::ifstream model_file(model_path);
      if (!models_list_file) {
        spdlog::error("Error opening file: {}", model_path);
        exit(EXIT_FAILURE);
      }
      // 输入trace
      std::string input_trace = "input.csv";
      cmd_parser.set_if_defined("trace_file", &input_trace);
      model_config["trace_file"] = input_trace;

      // 定义要处理的模型
      json model_json = json::parse(model_file);
      // make_unique 创建智能指针，引用LanguageModel::LanguageModel()构造LanguageModel
      auto model = std::make_unique<LanguageModel>(model_json, config, model_name);
      spdlog::info("Register Language Model: {}", model_name);
      simulator->register_language_model(model_config, std::move(model));
    }
    else {
      std::string model_name = model_config["name"];
      std::string onnx_path =
          fmt::format("{}/{}/{}.onnx", model_base_path, model_name, model_name);
      std::string mapping_path = fmt::format("{}/{}/{}.mapping", model_base_path,
                                            model_name, model_name);
      MappingTable mapping_table = MappingTable::parse_mapping_file(mapping_path, config);

      auto model = std::make_unique<Model>(onnx_path, model_config, config, model_name, mapping_table);
      spdlog::info("Register model: {}", model_name);
      simulator->register_model(std::move(model));
    }
  }
  simulator->run_simulator();

  /* Simulation time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("Simulation time: {:2f} seconds", duration.count());
  spdlog::info("Total tile: {}, simulated tile per seconds(TPS): {:3f}",
    simulator->get_number_tile(), simulator->get_tile_ops());
  return 0;
}

/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/
// 命令行参数解析器
#include "CommandLineParser.h"

namespace po = boost::program_options;

void CommandLineParser::parse(int argc, char** argv) noexcept(false) {
  // 解析命令行参数，将解析结果存储到variables_map中
  po::store(po::parse_command_line(argc, argv, options_description),
            variables_map);
  // 通知 variables_map，使其生效
  po::notify(variables_map);
}

void CommandLineParser::print_help_message_if_required() const noexcept {
  // 检查是否存在help选项
  if (variables_map.count("help") > 0) {
    // 打印所有命令行选项的描述信息，并退出程序
    std::cout << options_description << std::endl;
    exit(0);
  }
}

#include "BiasGelu.h"
#include "../Model.h"
// #include "../models/LanguageModel.h"

// 输入为onnx的激活函数构造
BiasGelu::BiasGelu(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {

    /* Load weight info from node */
    _input_shape = get_input(0)->get_dims();
    _bias_shape = get_input(1)->get_dims();

    assert(_input_shape.size()==3);
    _batch_size = _input_shape.at(0);
    _seq = _input_shape.at(1);
    _dk = _input_shape.at(2);

    _output_shape = _input_shape;
    Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(0));
    if (pre_defind_tensor == nullptr) {
        std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
            _id, node_proto.output(0), _output_shape, _config.precision, false);
            _outputs.push_back(output_tensor.get()->get_id());
        _model->add_tensor(std::move(output_tensor));
    } else {
        pre_defind_tensor->redefine_tensor(_id, _output_shape);
    }
    calculate_loops();
}

// 输入为模型配置的激活函数构造，没有使用
// std::map<std::string, std::string>在languagemodel.cc中存在
BiasGelu::BiasGelu(SimulationConfig config, Model* model,
               std::string name, std::map<std::string, std::string> &attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
//TODO:implement this

        /* Load weight info from model config */
        // _input_shape = parse_dims(get_attribute("input_shape"));
        // _bias_shape = parse_dims(get_attribute("weight_shape"));
        // _batch_size = 1;
        // int _nh = std::stoi(get_attribute("num_heads"));
        // int _dmodel = std::stoi(get_attribute("hidden_size"));
        // _dk = _dmodel / _nh;
        // _output_shape = _input_shape;
        // calculate_loops();
}

void BiasGelu::initialize_tiles(MappingTable& mapping_table) {
    for (uint32_t tokens= 0; tokens<_seq*_batch_size; tokens+=_tokens_per_tile) {
        uint32_t remain_tokens = std::min(_seq*_batch_size-tokens, _tokens_per_tile);
        std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = get_name(),
            .layer_id = _id,
            .accum = false,
        });
        /* dummy mapping */
        Mapping mapping;
        _tiles.push_back(std::move(tile));
        initialize_instructions(_tiles.back().get(), mapping, tokens, remain_tokens);

    }
}

void BiasGelu::initialize_instructions(Tile* tile, Mapping mapping, uint32_t token_offset, uint32_t tokens) {
    addr_type sram_base = SPAD_BASE;
    addr_type sram_bias_base = sram_base + tokens * _dk * _config.precision;

    addr_type first_addr, second_addr, output_addr;
    first_addr = get_operand_addr(_INPUT_OPERAND);
    second_addr = get_operand_addr(_INPUT_OPERAND+1);
    output_addr = get_operand_addr(_OUTPUT_OPERAND);
    /* Load two tile (input: tokens x _dk, skip: tokens x _dk) */
    std::set<addr_type> dram_addrs;
    std::set<addr_type> dram_output_addrs;
    std::set<addr_type> dram_skip_addrs;
    for (int offset=0; offset<tokens*_dk*_config.precision; offset+=_config.dram_req_size) {
        dram_addrs.insert(first_addr + token_offset*_dk*_config.precision + offset);
        dram_output_addrs.insert(output_addr + token_offset*_dk*_config.precision + offset);
    }

    for (int offset=0;offset<_dk*_config.precision; offset+=_config.dram_req_size)
        dram_skip_addrs.insert(second_addr + _seq*_dk*_config.precision+ offset);


    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_base,
        .size = (uint32_t)dram_addrs.size(),
        .src_addrs = std::vector<addr_type>(dram_addrs.begin(), dram_addrs.end()),
        .operand_id = _INPUT_OPERAND,  // query
    }));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_bias_base,
        .size = (uint32_t)dram_skip_addrs.size(),
        .src_addrs = std::vector<addr_type>(dram_skip_addrs.begin(), dram_skip_addrs.end()),
        .operand_id = _INPUT_OPERAND+1,  // query
    }));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .dest_addr = sram_base,
        .size = _dk * tokens * _config.precision / _config.dram_req_size,
        .compute_size = _dk * tokens * _config.precision,
        .src_addrs = std::vector<addr_type>{sram_base, sram_bias_base},
    }));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GELU,
        .dest_addr = sram_base,
        .size = _dk * tokens * _config.precision / _config.dram_req_size,
        .compute_size = _dk * tokens * _config.precision,
        .src_addrs = std::vector<addr_type>{sram_base},
    }));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVOUT,
        .dest_addr = sram_base,
        .size = (uint32_t)dram_output_addrs.size(),
        .src_addrs = std::vector<addr_type>(dram_output_addrs.begin(), dram_output_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
    }));
}

// SRAM双缓冲技术，将存储分一半，实现计算和预加载的重叠，提高并行性
void BiasGelu::calculate_loops() {
    // 每个token占用的字节数，dk是k的维度
    uint32_t size_per_token = _dk * _config.precision;
    // SRAM的容量
    uint32_t sram_capacity = _config.core_config[target_core].spad_size KB / 2;  // unit: byte

    _tokens_per_tile = (sram_capacity / size_per_token) - 1; 
    assert (_tokens_per_tile >= 1);
    if (_tokens_per_tile > _seq * _batch_size) _tokens_per_tile = _seq * _batch_size;

    spdlog::info("[BiasGeLU] tokens_per_tile: {}", _tokens_per_tile);
}
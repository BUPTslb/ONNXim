#可用参数
--config ：硬件配置文件的路径
--models_list ：模型文件的路径
--log_level ：设置日志的级别
--mode ：选择模型的类型
--trace_file ：语言模型的输入trace
--model : 模型的具体名称，用这个来生成onnx文件，暂时不需要
#参数说明
example/language_model.json: --models_list=model_path=model_file=model_json

# 128*128 systolic array resnet18
# booksim2
./build/bin/Simulator --config \
    ./configs/systolic_ws_128x128_c4_booksim2_tpuv4.json --model \
    ./example/models_list.json > results/ws_128x128_c4_booksim2_resnet18.txt

./build/bin/Simulator --config \
    ./configs/gemmini.json --model \
    ./example/models_list.json 

# remulator2
./build/bin/Simulator --config \
    ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2.json --model \
    ./example/models_list.json > results/ws_128x128_c4_simple_noc_half_ramulator2_resnet18.txt

# partition_quad
./build/bin/Simulator --config \
    ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_partition_quad.json --model \
    ./example/models_list.json > results/ws_128x128_c4_simple_noc_tpuv4_partition_quad_resnet18.txt

# simple_noc
./build/bin/Simulator --config \
    ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json --model \
    ./example/models_list.json > results/ws_128x128_c4_simple_noc_tpuv4_resnet18.txt


# 8*8 systolic array resnet18
./build/bin/Simulator --config \
    ./configs/systolic_ws_8x8_c1_booksim2_transformer.json --model \
    ./example/models_list.json > results/ws_8x8_c1_booksim2_transformer_resnet18.txt

./build/bin/Simulator --config \
    ./configs/systolic_ws_8x8_c1_simple_noc_transformer.json --model \
    ./example/models_list.json > results/ws_8x8_c1_simple_noc_transformer_resnet18.txt

./build/bin/Simulator --config \
    ./configs/systolic_ws_8x8_c4_booksim2_transformer.json --model \
    ./example/models_list.json > results/ws_8x8_c4_booksim2_transformer_resnet18.txt

./build/bin/Simulator --config \
    ./configs/systolic_ws_8x8_c4_simple_noc_transformer.json --model \
    ./example/models_list.json > results/ws_8x8_c4_simple_noc_transformer_resnet18.txt

example/language_models.json
## OPT-125M
# 128*128 systolic array resnet18
# booksim2
./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/systolic_ws_128x128_c4_booksim2_tpuv4.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_128x128_c4_booksim2_OPT125.txt

# remulator2
./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_half_ramulator2.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_128x128_c4_simple_noc_half_ramulator2_OPT125.txt

# partition_quad
./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4_partition_quad.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_128x128_c4_simple_noc_tpuv4_partition_quad_OPT125.txt

# simple_noc
./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_128x128_c4_simple_noc_tpuv4_OPT125.txt


# 8*8 systolic array resnet18
./build/bin/Simulator --trace_file input.csv --mode language  --config \
    ./configs/systolic_ws_8x8_c1_booksim2_transformer.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_8x8_c1_booksim2_transformer_OPT125.txt

./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/systolic_ws_8x8_c1_simple_noc_transformer.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_8x8_c1_simple_noc_transformer_OPT125.txt

./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/systolic_ws_8x8_c4_booksim2_transformer.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_8x8_c4_booksim2_transformer_OPT125.txt

./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/systolic_ws_8x8_c4_simple_noc_transformer.json --models_list \
    ./example/language_models.json > results/OPT-125M/ws_8x8_c4_simple_noc_transformer_OPT125.txt

#GEMMINI gemmini.json
./build/bin/Simulator --trace_file input.csv --mode language  --config \
    ./configs/gemmini.json --models_list \
    ./example/language_models.json > results/OPT-125M/gemmini-OPT125.txt

./build/bin/Simulator --trace_file input.csv --mode language  --config \
    ./configs/gemmini.json --models_list \
    ./example/language_models.json > results/OPT-125M/gemmini-HB-OPT125.txt

./build/bin/Simulator --trace_file input.csv --mode language --config \
    ./configs/gemmini.json --models_list \
    ./example/llama.json > results/llama-8b/gemmini-llama-8b.txt


import os
import json

# model_configs = ["v0", "v00", "v000", "v0000", "v00000", "v000000", "v0000000"]
model_configs = ["v1", "v11", "v111"]
seeds = ["1984", "2301", "3906", "4918"]

src_path = "/data3/tyx/mos-finetune-ssl/answer/iscslp2/metrics"

for config in model_configs:
    total_utterance = {
        "MSE": 0,
        "LCC": 0,
        "SRCC": 0,
        "KTAU": 0
    }
    total_system = {
        "MSE": 0,
        "LCC": 0,
        "SRCC": 0,
        "KTAU": 0
    }

    for seed in seeds:
        fn = f"{config}_{seed}.json"
        with open(os.path.join(src_path, fn), "r") as f:
            data = json.load(f)

        for key in total_utterance:
            total_utterance[key] += data['utterance'][key]
                
        for key in total_system:
            total_system[key] += data['system'][key]

    num_files = len(seeds)
    average_utterance = {key: total_utterance[key] / num_files for key in total_utterance}
    average_system = {key: total_system[key] / num_files for key in total_system}
    total_dict = {
        "utterance": average_utterance,
        "system": average_system,
    }
    total_json = json.dumps(total_dict, indent=4)
    print(f'config: {config}\n{total_json}\n')
    with open(os.path.join(src_path, f"{config}.json"), "w") as f:
        f.write(total_json)
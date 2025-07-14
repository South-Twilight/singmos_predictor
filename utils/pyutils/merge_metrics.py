import os
import json
import argparse

def extract_info_from_path(base_info):
    items = base_info.split('_BASE_')
    pretrain = ''
    finetune = ''
    if 'BASE' in base_info:
        # e.g.: ft_singmos_full_finetune.v1_BASE_pt_singmos_v1_pretrain.v1
        pretrain = "_".join(items[1].split("_")[1:3]) 
        finetune = "_".join(items[0].split("_")[1:3]) 
    else:
        # e.g.: pt_singmos_v1_pretrain.v1
        pretrain = "_".join(items[0].split("_")[1:3]) 

    return pretrain, finetune


def main(args):
    pretrain, finetune = extract_info_from_path(args.src_dir)
    for eval_set in args.eval_sets:
        for eval_tp in args.eval_types:
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
            ans_dir = os.path.join(args.src_dir, eval_set, eval_tp)
            for rd_dir in args.random_seeds:
                json_path = os.path.join(ans_dir, f"rand_{rd_dir}", "metrics.json")
                with open(json_path, "r") as f:
                    data = json.load(f)
                for key in total_utterance:
                    total_utterance[key] += data['utterance'][key]
                for key in total_system:
                    total_system[key] += data['system'][key]

            num_files = len(args.random_seeds)
            average_utterance = {key: total_utterance[key] / num_files for key in total_utterance}
            average_system = {key: total_system[key] / num_files for key in total_system}
            total_dict = {
                "system": average_system,
                "utterance": average_utterance,
            }
            total_json = json.dumps(total_dict, indent=4)
            with open(os.path.join(args.src_dir, "metrics.json"), "w") as f:
                f.write(total_json)
            print(f'{os.path.join(args.src_dir, "metrics.json")}: \n{total_json}\n')


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge answers with different random seed."
    )
    parser.add_argument("--src_dir", help="directory of source answers with different random seed.")
    parser.add_argument(
        "--random_seeds", 
        nargs='+', type=int, default=[0, 1234, 1984],
        help="list of random seeds."
    )
    parser.add_argument(
        "--eval_sets", 
        nargs='+', type=str, default=["singmos_v1", "singmos_v2"],
        help="list of evaluation sets."
    )
    parser.add_argument(
        "--eval_types", 
        nargs='+', type=str, default=["ALL", "ID", "OOD"],
        help="list of types for evaluation sets."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)
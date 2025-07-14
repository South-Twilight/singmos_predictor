import os
import json
import pandas as pd
import argparse

metrics_order = ["MSE", "LCC", "SRCC", "KTAU"]

def parse_args():
    parser = argparse.ArgumentParser(description="Process metrics and generate a formatted CSV table.")
    parser.add_argument("--base_path", required=True, help="Base path for metrics files.")
    parser.add_argument("--output_file", required=True, help="Output CSV file path.")
    parser.add_argument(
        "--pt_datasets", type=str, nargs='+',
        default=["singmos_v1", "singmos_v2", "singmos_full"],
        help="List of pretrain datasets."
    )
    parser.add_argument(
        "--ft_datasets", type=str, nargs='+',
        default=["-", "singmos_full"],
        help="List of finetune datasets."
    )
    parser.add_argument(
        "--test_sets", type=str, nargs='+',
        default=["singmos_v1", "singmos_v2"],
        help="List of test sets."
    )
    parser.add_argument(
        "--splits", type=str, nargs='+',
        default=["ALL", "ID", "OOD"],
        help="List of split sets."
    )
    return parser.parse_args()


def main(args):
    pt_datasets = args.pt_datasets
    ft_datasets = args.ft_datasets
    test_sets = args.test_sets
    splits = args.splits
    base_path = args.base_path
    output_file = args.output_file

    rows = []

    for pt_dataset in pt_datasets:
        for ft_dataset in ft_datasets:
            if ft_dataset == "-":
                ans_dir = os.path.join(base_path, f"pt_{pt_dataset}_pretrain.v1")
            else:
                ans_dir = os.path.join(base_path, f"ft_{ft_dataset}_finetune.v1_BASE_pt_{pt_dataset}_pretrain.v1")
            for test_set in test_sets:
                for split_set in splits:
                    metric_file = os.path.join(ans_dir, test_set, split_set, "metrics.json")
                    if not os.path.exists(metric_file):
                        print(f"Warning: {metric_file} does not exist. Skipping.")
                        continue
                    with open(metric_file, "r") as f:
                        data = json.load(f)
                    base_info = {
                        "Pretrain Dataset": pt_dataset,
                        "Finetune Dataset": ft_dataset,
                        "Test Set": test_set,
                        "Split": split_set,
                    }
                    system_metrics = {f"Sys.{k}": v for k, v in data["system"].items()}
                    utterance_metrics = {f"Utt.{k}": v for k, v in data["utterance"].items()}
                    row = {**base_info, **system_metrics, **utterance_metrics}
                    rows.append(row)

    df = pd.DataFrame(rows)
    info_cols = ["Pretrain Dataset", "Finetune Dataset", "Test Set", "Split"]
    metric_cols = [col for col in df.columns if col not in info_cols]
    df = df[info_cols + metric_cols]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保输出目录存在
    df.to_csv(output_file, index=False)
    print(f"Formatted table saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
"""
Quick config sanity check for all original task ids in SeldNet_3classes_80.
"""

import parameters


TASK_IDS = ["1", "2", "3", "4", "5", "6", "7", "9", "10", "32", "33", "34", "333", "999"]


def main():
    print("task_id | label_hop | label_seq | feat_seq | t_pool_size | model | dataset")
    print("-" * 88)
    for task_id in TASK_IDS:
        params = parameters.get_params(task_id)
        print(
            f"{task_id:>6} | "
            f"{params['label_hop_len_s']:<9} | "
            f"{params['label_sequence_length']:<9} | "
            f"{params['feature_sequence_length']:<8} | "
            f"{str(params['t_pool_size']):<11} | "
            f"{params['model']:<8} | "
            f"{params['dataset']}"
        )


if __name__ == "__main__":
    main()

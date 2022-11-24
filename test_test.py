from slicing.dataset_splitting import create_dataset_flower, create_dataloader_from_dataset_flower
import numpy as np
import time

def main():
    train_dataset, val_dataset, test_dataset = create_dataset_flower(
        "../Processed/Sliced",
        640,
        16,
        1,
        cache=False,
        num_files=1000
    ).split_data(
        proportions=[80, 16, 4],
        stratification_level=1
    )

    # def get_classes(obj):
    #     classes = [type(obj)]
    #     while isinstance(obj, list) and len(obj) > 0:
    #         classes.append(type(obj[0]))
    #         obj = obj[0]
            
    #     return classes

    # for i, t in zip([train_dataset, val_dataset, test_dataset], ["train", "validation", "test"]):
    #     print("-----------------", t, "=", len(i), "-----------------\n")
    #     for k, v in i.dataset.__dict__.items():
    #         if not isinstance(v, str) and hasattr(v, "__len__"):
    #             print(k, f'({get_classes(v)})', ":", len(v))
    #         else:
    #             print(k, f'({get_classes(v)})', "=", v)
    start = time.process_time()
    dataloader = create_dataloader_from_dataset_flower(train_dataset, 16)
    print("Dataloader initialization time :", time.process_time() - start)

if __name__ == '__main__':
    main()
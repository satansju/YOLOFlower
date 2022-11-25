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
        num_files=None,
        min_items=2
    ).split_data(
        proportions=[80, 16, 4],
        stratification_level=1
    )

    start = time.process_time()
    dataloader = create_dataloader_from_dataset_flower(train_dataset, 16, gamma = 1)
    print("Dataloader initialization time :", time.process_time() - start)

if __name__ == '__main__':
    main()
    
    
    # Debugging plots for the weighting function 
    # (should be put inside the create_dataloader_from_dataset_flower function after the weights have been calculated)
    # fig = plt.hist(weights)
    # plt.savefig("../test.png")
    # plt.close()
    # fig = plt.scatter(weights, [len(i) for i in dataset.labels])
    # plt.savefig("../test1.png")
    # plt.close()
    # fig = plt.scatter(weights, [np.unique(i[:, 0]).shape[0] for i in dataset.labels])
    # plt.savefig("../test2.png")
    # plt.close()
    # fig = plt.bar(np.arange(0, 6, 1), np.bincount([np.unique(i[:, 0]).shape[0] for i in dataset.labels], weights=weights, minlength=5))
    # plt.savefig("../test3.png")
    # plt.close()
    # fig = plt.bar(np.arange(0, 6, 1), np.bincount([np.unique(i[:, 0]).shape[0] for i in dataset.labels], minlength=5))
    # plt.savefig("../test4.png")
    # return
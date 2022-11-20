import os

import pandas as pd
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from typing import Dict, List, Tuple

from utils.dataloaders import * 


class LoadFlower(LoadImagesAndLabels):
    def __init__(self, *args, **kwargs):
        if "simple" in kwargs:
            return        
        kwargs = {k : v for k, v in kwargs.items() if not k == "simple"}
        super().__init__(*args, **kwargs)
        path_tree = pd.concat([pd.DataFrame({level: [name]
                              for level, name in enumerate(i.rsplit(os.sep))}) for i in self.im_files])
        path_tree_num_unique = path_tree.agg(lambda x: len({i for i in x}))
        self.path_tree = path_tree.loc[:, path_tree_num_unique > 1]
        print(args)
        self.batch_size = args[1]

    def split_data(self, proportions: List[float] or List[int] = [80, 16, 4], stratification_level: int = 1) -> List[Subset]:
        proportions = [i / sum(proportions) for i in proportions]

        def group_split(ind, proportions, groups):
            group_set = {i for i in groups}

            n = len(group_set)
            splits = [max(int(n * i), 1) for i in proportions]
            left = n - sum(splits)
            assert left >= 0, print("Number of splits must be smaller than or equal to number of groups.")
            
            for i in range(left):
                splits[i] += 1

            try:
                assert n == sum(splits)
            except:
                raise ValueError(splits)

            group_splits = []
            for i in splits:
                this_split = set(random.sample(group_set, i))
                group_splits.append(this_split)
                group_set = group_set - this_split

            assert len(group_set) == 0

            return [[ind for g, ind in zip(groups, ind) if g in i] for i in group_splits]

        splits = group_split(
            ind=range(len(self)),
            proportions=proportions,
            groups=self.path_tree.iloc[:, :stratification_level].apply(
                lambda x: ','.join(x.dropna().astype(str)), axis=1)
        )

        datasets = [Subset(self, ind) for ind in splits]
        [setattr(dataset, "dataset", self.__subset_attr(ind)) for dataset, ind in zip(datasets, splits)]
        
        print(datasets[0].dataset)

        return tuple([*datasets])
    
    def __subset_attr(self, ind):
        n = len(self)
        
        sub = LoadFlower(simple = True)
        
        for k, v in self.__dict__.items():
            if hasattr(v, "__len__") and len(v) == n:
                if isinstance(v, pd.DataFrame):
                    setattr(sub, k, v.iloc[ind,:])
                elif isinstance(v, np.ndarray):
                    setattr(sub, k, v[ind])
                else:
                    try:
                        setattr(sub, k, [v[i] for i in ind])
                    except:
                        raise ValueError("Length of", k, "is", len(k))
            else:
                setattr(sub, k, v)
                
        return sub

        # bi = np.floor(np.arange(len(ind)) / self.batch_size).astype(int)  # batch index
        # self.batch = bi  # batch index of image
        # self.n = n
        # self.indices = range(self.n)


def create_dataset_flower(path,
                          imgsz,
                          batch_size,
                          stride,
                          augment=False,  # augmentation
                          hyp=None,  # hyperparameters
                          rect=False,  # rectangular batches
                          cache=False,
                          single_cls=False,
                          pad=0.0,
                          image_weights=False,
                          prefix='',
                          rank=-1):
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadFlower(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    return dataset


def create_dataloader_from_dataset_flower(dataset,
                                          batch_size,
                                          rank=-1,
                                          workers=8,
                                          image_weights=False,
                                          quad=False,
                                          shuffle=False):
    batch_size = min(batch_size, len(dataset))

    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)

    weights = [1 for i in range(len(dataset))]

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(
        dataset), generator=generator) if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)

    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset
import random
# from math import round

def split_data(ind, proportions, groups):
    proportions = [i/sum(proportions) for i in proportions]

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

        # for i in group_splits:
        #     print(groups)

        return [[ind for g, ind in zip(groups, ind) if g in i] for i in group_splits]

    return group_split(
        ind=range(len(ind)),
        proportions=proportions,
        groups = groups
        )

print(split_data(list(range(100)), [80, 16, 4], [i for i in range(10) for j in range(10)]))
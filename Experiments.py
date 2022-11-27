from train import run
import re

for gamma in [0, .5, 1, 1.5, 2]:
    run(
        img=640, 
        batch=128, 
        epochs=10, 
        workers=8,
        optimizer="AdamW", 
        data="data/Flower.yaml", 
        hyp="data/hyps/FlowerHyp.yaml", 
        cache="RAM", 
        name=re.sub("\.", "_", f'test_exp_{gamma}_test'),
        gamma=float(gamma),
        min_items=0,
        dataset_size=100
    )
    print("TEST")
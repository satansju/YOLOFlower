from train import run
import re

run(
    img=640, 
    batch_size=32, 
    epochs=10, 
    workers=0,
    optimizer="AdamW", 
    data="data/Flower.yaml", 
    hyp="data/hyps/FlowerHyp.yaml", 
    cache="RAM", 
    name="test_new_files",
    gamma=2,
    class_weights=True,
    deterministic=True,
    dataset_size=500
)



# if "a":
#     raise NotImplementedError("Remember to implement resizing dataset during experiment!!!!")

# run(
#     img=640, 
#     batch_size=32, 
#     epochs=10, 
#     workers=0,
#     optimizer="AdamW", 
#     data="data/Flower.yaml", 
#     hyp="data/hyps/FlowerHyp.yaml", 
#     cache="RAM", 
#     name="baseline",
#     deterministic=True
# )

# run(
#     img=640, 
#     batch_size=32, 
#     epochs=10, 
#     workers=0,
#     optimizer="AdamW", 
#     data="data/Flower.yaml", 
#     hyp="data/hyps/FlowerHyp.yaml", 
#     cache="RAM", 
#     name="testing_loss_gamma_1",
#     gamma=1,
#     deterministic=True
# )

# run(
#     img=640, 
#     batch_size=32, 
#     epochs=10, 
#     workers=0,
#     optimizer="AdamW", 
#     data="data/Flower.yaml", 
#     hyp="data/hyps/FlowerHyp.yaml", 
#     cache="RAM", 
#     name="testing_loss_gamma_1_flgamma_1d5",
#     gamma=1,
#     deterministic=True
# )

# run(
#     img=640, 
#     batch_size=32, 
#     epochs=10, 
#     workers=0,
#     optimizer="AdamW", 
#     data="data/Flower.yaml", 
#     hyp="data/hyps/FlowerHyp.yaml", 
#     cache="RAM", 
#     name="testing_loss_gamma_1_flgamma_1d5",
#     gamma=1,
#     deterministic=True
# )


# for gamma in [0, .5, 1, 1.5, 2]:
#     for cv in range(3):
#         run(
#             img=640, 
#             batch_size=32, 
#             epochs=10, 
#             workers=0,
#             optimizer="AdamW", 
#             data="data/Flower.yaml", 
#             hyp="data/hyps/FlowerHyp.yaml", 
#             cache="RAM", 
#             name=re.sub("\.", "d", f'new_gamma_{gamma}_cv{cv}'),
#             gamma=float(gamma),
#             deterministic=False
#         )
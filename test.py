from train import run

run(
    img=640,
    batch_size=32,
    epochs=8,
    workers=0,
    optimizer="AdamW",
    data="data/Flower.yaml",
    hyp="data/hyps/FlowerHyp.yaml",
    cache="RAM",
    name=f'test_run',
    deterministic=True,
    weights="yolov5s.pt",
    gamma=1,
    class_weights=True,
    dataset_size=500
)
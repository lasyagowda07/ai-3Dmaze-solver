import torch
from model import DQN

PT_PATH = "checkpoints/maze3d_dqn.pt"
ONNX_PATH = "maze_dqn3d.onnx"


def main():
    device = "cpu"
    model = DQN(input_dim=18, hidden=256, output_dim=6).to(device)
    model.load_state_dict(torch.load(PT_PATH, map_location=device))
    model.eval()

    dummy = torch.randn(1, 18, device=device)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["obs"],
        output_names=["q"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "q": {0: "batch"}},
    )

    print("exported:", ONNX_PATH)


if __name__ == "__main__":
    main()
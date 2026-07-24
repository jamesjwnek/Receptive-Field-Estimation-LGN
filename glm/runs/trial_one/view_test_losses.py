from pathlib import Path

directory_path = Path.cwd()

run_files = [file.name for file in directory_path.glob("*.json")]

import json

for name in run_files:
    with open (name, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Test loss: {data["test_loss"]}, Stride x: {data["stride_val_x"]}, Stride y: {data["stride_val_y"]}")
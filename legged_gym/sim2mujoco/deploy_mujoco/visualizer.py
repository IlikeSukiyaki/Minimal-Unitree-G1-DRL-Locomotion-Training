#!/usr/bin/env python3
"""
visualizer.py: MuJoCo XML model viewer using the official Python bindings.

Usage:
    python visualizer.py /path/to/your_model.xml

Requirements:
    pip install mujoco
"""
import sys
import mujoco
import mujoco.viewer

def main(xml_path: str):
    # Load model and create data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Launch blocking viewer: steps physics and renders in one call
    mujoco.viewer.launch(model, data)  # managed viewer mode :contentReference[oaicite:0]{index=0}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualizer.py /path/to/your_model.xml")
        sys.exit(1)
    main(sys.argv[1])

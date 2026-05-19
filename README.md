# Isaac Sim-Based Robotic Manipulation & Data Collection Pipeline

This project implements a simulation-based robotic manipulation pipeline using a Franka Panda arm in NVIDIA Isaac Sim, focusing on pick-and-place tasks and synthetic data generation for Vision-Language-Action (VLA) workflows.

## 🎥 Demo

[Watch Full Video](pick_place.mp4)

## 🚀 Overview

The system enables autonomous pick-and-place execution in simulation by combining object-centered control, trajectory-based motion, and gripper actuation. It is designed to generate structured interaction data for learning-based robotic systems.

## 🧠 Key Features

- Franka Panda manipulation in Isaac Sim
- Object-centered end-effector positioning
- Closed-loop manipulation via state-machine logic
- Trajectory-based motion execution
- Gripper control for pick-and-place
- Synthetic data generation for VLA models

## 🏗️ Pipeline

Object Pose → Target Pose → Trajectory → Execution → Data Collection

## ⚙️ Tech Stack

- NVIDIA Isaac Sim
- Python
- Robotics Toolbox for Python
- NumPy

## ▶️ Run

```bash
./python.sh main.py

# Cricket-Bowling-Biomechanics
# Cricket Bowling Analysis

A computer vision–based project for analyzing **cricket bowling biomechanics** from single-camera video using pose estimation, trajectory analysis, and temporal alignment.

This repository focuses on extracting **wrist and elbow kinematics** from side-view bowling videos and studying motion patterns, limitations, and biomechanical signals such as trajectories, angles, and timing.

---

## 🚀 Project Overview

Cricket bowling is a fast, highly dynamic motion involving complex joint coordination. Traditional biomechanical analysis requires expensive motion-capture setups. This project explores how far we can go using **only a single RGB video** and modern pose estimation models.

The pipeline:

```
Input Video
   ↓
Pose Estimation (MediaPipe)
   ↓
Wrist & Elbow Keypoints
   ↓
Trajectory Generation
   ↓
Temporal Alignment (DTW)
   ↓
Biomechanics Analysis
```

---

## ✨ Features

* 🎯 Wrist & elbow keypoint extraction from video
* 📈 2D trajectory generation for the bowling arm
* ⏱️ High-FPS processing (tested at 50 FPS)
* 🔁 Temporal alignment of deliveries using FastDTW
* 🧠 Biomechanics-aware analysis pipeline
* 🎥 Trajectory overlay visualization on video

---

## ⚠️ Known Limitations

* **Self-occlusion**: When the bowling hand passes behind the leg, wrist keypoints become noisy due to loss of visual evidence.
* **Single-view constraint**: Depth and out-of-plane motion are limited with monocular input.
* **Pose jitter**: Raw pose estimates require smoothing for reliable biomechanics.

These limitations are explicitly studied and addressed through filtering and temporal modeling.

---

## 🛠️ Tech Stack

* **Python 3.10**
* **MediaPipe Pose** – pose estimation
* **OpenCV** – video I/O and visualization
* **NumPy / SciPy** – numerical computation
* **FilterPy** – Kalman filtering (smoothing)
* **FastDTW** – temporal alignment of trajectories
* **Matplotlib** – plotting and analysis

---

## 📁 Project Structure

```
cricket-bowling-analysis/
│
├── main.py                    # Entry point
├── src/
│   └── pose_estimator.py      # Pose estimation & keypoint extraction
│
├── utils/
│   └── video_utils.py         # Video read/write helpers
│
├── input_video/               # Input bowling videos
├── output_video/              # Trajectory overlay outputs
├── requirements.txt
└── README.md
```

---

## ▶️ Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/cricket-bowling-analysis.git
cd cricket-bowling-analysis
```

### 2️⃣ Create environment (recommended)

```bash
conda create -n cricketproject python=3.10
conda activate cricketproject
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the pipeline

```bash
python main.py
```

Output videos with trajectory overlays will be saved in `output_video/`.

---

## 📊 Example Output

* Wrist trajectory plotted over time
* Trajectory overlay on bowling video
* Temporally aligned deliveries using DTW

*(Sample visuals to be added)*

---

## 🔬 Future Work

* Velocity & acceleration profiling
* Elbow angle and extension analysis
* Ball release frame detection
* Visibility-aware and biomechanical filtering
* Multi-view extension (if data available)
* Open dataset for cricket bowling biomechanics

---

## 📚 Motivation

This project is built from **first principles**, aiming to understand:

* What pose models can and cannot infer in fast sports motions
* How occlusion affects biomechanical signals
* How far single-camera vision can go in sports analytics

It is both a **learning project** and a foundation for more rigorous sports biomechanics research.

---

## 🤝 Contributions

Contributions, ideas, and discussions are welcome.

If you are interested in:

* Sports analytics
* Biomechanics
* Computer vision for human motion

Feel free to open an issue or reach out.

---

## 📜 License

MIT License (to be updated if needed)

---

## ⭐ Acknowledgements

* MediaPipe team for pose estimation
* Open-source CV & biomechanics community

---

*Built with curiosity, rigor, and a love for cricket 🏏*

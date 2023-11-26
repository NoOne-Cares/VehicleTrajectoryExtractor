
# Vehicle Trajectory Analysis using OpenCV and YOLOv8

## Project Overview
Modern transportation systems often require advanced tools for analyzing vehicle movement patterns, and this Python project aims to provide a solution using computer vision techniques. By combining the power of OpenCV for image processing and YOLOv8 for object detection, the project focuses on detecting Indian vehicles, extracting their trajectories, segregating trajectories based on vehicle types, and transforming these trajectories into a single plane using complex homology. The resulting data can be invaluable for further research and analysis in transportation studies.

## Objectives
* Vehicle Detection: Utilize the YOLOv8 model to detect vehicles in Indian road scenarios. YOLOv8 is chosen for its real-time object detection capabilities.

* Trajectory Extraction: Uses [Sort](https://github.com/abewley/sort) algorithm to extract vehicle trajectories from video footage. This involves tracking the movement of vehicles over time.

* Vehicle Segregation: Classify detected vehicles into different types (e.g., cars, buses, trucks , three wheelers) based on YOLOv8's classification outputs.

* Homology Transformation: Apply homology transformations to convert the extracted trajectories into a single plane. This process enhances the utility of trajectory data for various research purposes.

## How It Works

#### Vehicle Detection:
YOLOv8 is employed to perform object detection on each frame of the input video. It identifies vehicles and provides bounding box coordinates.

#### Trajectory Extraction:
The bounding box coordinates from consecutive frames are used to establish vehicle trajectories. [Sort](https://github.com/abewley/sort) algorithm is applied to connect vehicle positions over time.

#### Vehicle Segregation:
YOLOv8 provides not only object detection but also classification. This information is used to segregate trajectories based on the types of vehicles detected.

#### Homology Transformation:

Homology transformations are applied to transform the trajectories into a single plane. This ensures consistency in the representation of vehicle movement for further analysis.


## Applications
1. Traffic Flow Analysis: The extracted trajectories can be used to analyze the flow of traffic, identify congestion points, and optimize traffic management strategies.

2. Urban Planning: Understanding vehicle movement patterns is crucial for urban planners to make informed decisions about road infrastructure and public transportation systems.

3. Safety and Security: Analyzing vehicle trajectories can contribute to enhancing road safety measures and improving security monitoring on road networks.

## Prerequisites
Before running the project, ensure you have the following dependencies installed:

* [Python 3.10 or above](https://www.python.org/downloads/release/python-3109/)
* [venv](https://docs.python.org/3/library/venv.html)

## Project Structure

```
project-root/
|-- data/
|   |-- input/
|       |-- video.mp4
|   |-- weights/
|           |-- (YOLOv8 weights file )
|-- src/
|   |-- sort.py
|   |-- main.py
|-- results/
|-- README.md
|-- requirements.txt
```
## Usage
1. Create a [virtual enviroment](https://python.land/virtual-environments/virtualenv) using python Python 3.10 

2. Clone the repository:

```bash
git clone https://github.com/NoOne-Cares/VehicleTrajectoryExtractor.git
cd VehicleTrajectoryExtractor
```
2. Install dependencies:

```bash
Copy code
pip install -r requirements.txt
Place your input video file (video.mp4) in the data/input/ directory.
```
3. Replace the file name.extension in the main.py with the video file name
```python
vedio = "data/input/<filename.extension>"
```
4. Run the main.py file:

```bash
python src/main.py
```
5. Follow the on-screen instructions to perform trajectory extraction, vehicle segregation, and homology transformation.

6. View the results in the results/ directory.

## Future Improvements
* Improve Accuracy: Improve the accuracy of the model for Indian road by expanding the data set with more number of images and more variety of Indian vehicles.

* Semantic Segmentation: Enhance vehicle segregation by incorporating semantic segmentation techniques to classify vehicles more accurately.

* Integration with GIS Data: Integrate trajectory data with Geographic Information System (GIS) data for a more comprehensive analysis of vehicle movement in a spatial context.

* Real-Time Analysis: Optimize the algorithms for real-time performance, enabling live analysis of traffic scenarios.

## Acknowledgements

 - [YOLO V8](https://docs.ultralytics.com/)
 - [SORT](https://github.com/abewley/sort)














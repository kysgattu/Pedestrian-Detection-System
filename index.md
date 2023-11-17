# Pedestrian Detection System using YOLO
The Pedestrian Detection System using YOLO is an innovative project designed to assist the Campus Planning Department at Facilities Management at the University of Massachusetts Lowell. The primary objective of this initiative is to create a tool that leverages advanced Machine Learning Techniques, specifically utilizing the potent YOLO (You Only Look Once) Deep Learning Algorithm developed by Ultralytics. The purpose of this tool is to accurately count the number of individuals traversing specific pathways across the university campus.

With a user-centric approach, we have crafted a sophisticated User Interface as an integral part of the tool. This interface enables users to effortlessly upload a video and define specific regions along the paths where pedestrian detection is required. The system seamlessly executes the process of detecting, tracking, and tallying the number of individuals traversing these designated pathways.

By incorporating YOLO, renowned for its efficiency in object detection tasks, our system ensures robust and real-time detection capabilities. This translates to a more streamlined and accurate monitoring process, enhancing the overall efficiency of the Campus Planning Department in managing pedestrian flow.

In summary, the Pedestrian Detection System utilizing YOLO stands as a testament to the integration of cutting-edge technology to address practical challenges. By providing a user-friendly interface and leveraging the power of YOLO, this project facilitates the precise and automated counting of pedestrians, thereby contributing to enhancing campus planning and management at the University of Massachusetts Lowell.


## Table of Contents
- [Prerequisites](#prerequisites)
    - [Environment](#environment)
    - [Video Description](#video-description)
- [Modules](#modules)
    - [User Interface](#tkmodule)
    - [Region of Interest Selector](#roi)
    - [YOLO Model](#yolo)
    - [Working of the Detection Code](#code)
    - [Result Visualisation](#visualisation)
    - [Result - Annotated Video](#result)
- [Working Demo](#demo)

- [Developers](#developers)
- [Links](#links)
- [References](#references)            

## Prerequisites <a name='prerequisites'></a>

### Environment <a name='environment'></a>

1. Python 3 Environment
2. Python modules required: NumPy, Pandas, PyTorch, Opencv2, Matplotlib, Ultralytics, Supervision, Tkinter, TQDM

OR
- Any Python3 IDE installed with the above modules. (Pycharm is used in the development of this project)

### Video Description <a name='video-description'></a>

The tool supports the usage of timelapse videos of any kind and any size with a good resolution and frame rate. Optimize your frame rate and quality of the video based on how big are the persons in the video(Smaller persons in the frame require the video to be clearer).
The video used during the development and testing of the model is a timelapse video of a place on the campus of UMass Lowell which contains multiple paths on which the number of people passing has to be counted.

##### Test Video Information 

- Height - 720
- Width - 1280
- Frame Rate - 30 FPS
- Timelapse Rate - 1 Hour/minute

https://github.com/kysgattu/Pedestrain-Detection-System/assets/42197976/7b929dc9-bdb4-4b71-9ef2-a800f3e86184

## Modules<a name='modules'></a>

> ### User Interface <a name = 'tkmodule'></a>

The Front-End UI is a Tkinter Dialog box where 
- A video file on which the person detection is to be performed can be browsed from the File System. 
- A Target location on the file system where the result annotated video has to be saved
- Number of Regions of Interest on which the detection has to be performed
- Names for each Region of Interest

<img width="1280" alt="Screenshot 2023-11-16 at 7 35 35 PM" src="https://github.com/kysgattu/Pedestrain-Detection-System/assets/42197976/aa10d33c-d507-4185-a20a-b716902849dc">


> ### Region Of Interest Selector <a name = 'roi'></a>

Once the Detection button is run, For each Region of Interest a Matplot with a frame from the video pops up on which four points have to be selected which encloses the detection region.

<img width="1280" alt="Screenshot 2023-11-16 at 7 37 57 PM" src="https://github.com/kysgattu/Pedestrain-Detection-System/assets/42197976/338cb935-27a8-442e-b462-c1c8b79763fd">


> ### YOLO Model <a name = 'yolo'></a>
YOLO, or You Only Look Once, is a popular computer-vision object detection algorithm. The key idea behind YOLO is speed and efficiency. Instead of dividing the image into a grid and running object detection on each grid cell, YOLO divides the image into a grid but performs detection for all objects within the entire image in one forward pass of the neural network. YOLO divides the input image into a grid. Each grid cell is responsible for predicting bounding boxes and class probabilities for the objects contained in that cell. Each grid cell predicts multiple bounding boxes along with confidence scores. These bounding boxes represent the location of potential objects in the image. YOLO also predicts the probability of the presence of different classes within each bounding box. After predicting multiple bounding boxes and class probabilities, YOLO uses non-maximum suppression to eliminate duplicate or low-confidence detections. This helps to provide a cleaner and more accurate set of predictions.

> ### Working of the Detection Code <a name = 'code'></a>

The code initiates by configuring parameters like confidence levels, scaling percentages, and tracking thresholds. The video is processed frame by frame using the OpenCV library, with optional scaling for enhanced performance. Regions of interest (ROIs) are defined within each frame, and the code iterates through these, applying YOLO to identify individuals. Subsequently, a tracking mechanism based on object centers is employed to trace the movement of detected persons across frames. The count of individuals within each ROI is continuously updated, and the annotated frames, showcasing bounding boxes, tracking information, and ROI overlays, are compiled into an output video. The final results, including the number of persons detected and tracked in each ROI, are presented upon completion.

> ### Results Visualisation <a name = 'visualisation'></a>

The annotated frames with bounding boxes, tracking information, and ROI overlays are stored in an output video and saved for later review. After processing the entire video, the code prints the number of persons detected and tracked in each ROI in the Tkinter Dialog Box.

<img width="1280" alt="Screenshot 2023-11-15 at 2 39 37 PM" src="https://github.com/kysgattu/Pedestrain-Detection-System/assets/42197976/4a53344a-4714-4553-bc82-a60788eb9d90">

> ### Result - Annotated Video <a name = 'result'> </a>
The Annotated video with the number of persons in each ROI is shown in the video - 

[Result Video](https://github.com/kysgattu/Pedestrain-Detection-System/assets/42197976/9d392884-4b89-49e3-8795-f750e2356f84)


## Working Demo of the Tool <a name = 'demo'></a>

##### Step-by-Step Instructions for running the tool

- Browse for the source video and the directory to save the annotated video.
- Specify the number of regions (paths) for detecting passing pedestrians.
- Provide names for each region, separated by commas (ensure the number of names matches the specified regions).
- Hit the "Start Detection" button.
- Press enter after selecting four points in the window displaying a frame from the video. Repeat this process for each region; the selection windows will pop up automatically based on the specified number of regions.
- Once the regions are selected, the detection process starts automatically. Note that the progress bar is visible only on the command line.
- After detection is complete, view the counts of each region in the results box.
- Keep in mind that only one detection can be performed per code execution. If errors occur or mistakes are made during data entry, restart the app.

*_A detailed demo of how the tool can be used is shown in below video:_*

[Demo Video](https://github.com/kysgattu/Pedestrain-Detection-System/assets/42197976/395e06fc-b10f-4f5c-80f1-ea486e98991b)


## Developers <a name='developers'></a>
* [Kamal Yeshodhar Shastry Gattu](https://github.com/kysgattu)

## Links <a name='links'></a>
> GitHub:[G K Y SHASTRY](https://github.com/kysgattu)
> 
> [University of Massachusetts Lowell](https://www.uml.edu/)
> 
> [Facilites Information Systems - Facilities Management - UMass Lowell](https://www.uml.edu/facilities/planning-design-construction/fis.aspx)
> 
> Contact me:     <gkyshastry0502@gmail.com> , <kysgattu0502@gmail.com>

## References <a name='references'></a>

> - [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://doi.org/10.48550/arXiv.2004.10934)
>  
> - [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo)
>  
> - [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)

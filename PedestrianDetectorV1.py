# File: PedestrianDetectorV1.py
# Author: Kamal Yeshodhar Shastry Gattu <KamalYeshodharShastry_Gattu@uml.edu> <kysgattu0502@gmail.com>
# Date: November 1, 2023
# Description: This is a tool that Detects, Tracks and Counts number of people passing across particular path(s) in a Timelapse video

# Pedestrian Detection
# !pip install -q -r requirements.txt

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import subprocess
from tqdm import tqdm
import supervision as sv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


# Display image and videos
# from IPython.display import Video, display
# %matplotlib inline

def resize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def filter_tracks(centers, patience):
    """Function to filter track history"""
    filter_dict = {}
    for k, i in centers.items():
        d_frames = i.items()
        filter_dict[k] = dict(list(d_frames)[-patience:])
    return filter_dict

def update_tracking(centers_old,obj_center, thr_centers, lastKey, frame, frame_max):
    """Function to update track of objects"""
    is_new = 0
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
    lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
    # Calculating distance from existing centers points
    previous_pos = [(k,obj_center) for k,centers in lastpos if (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]
    # if distance less than a threshold, it will update its positions
    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = obj_center
    # Else a new ID will be set to the given object
    else:
        if lastKey:
            last = lastKey.split('D')[1]
            id_obj = 'ID' + str(int(last)+1)
        else:
            id_obj = 'ID0'
        is_new = 1
        centers_old[id_obj] = {frame:obj_center}
        lastKey = list(centers_old.keys())[-1]
    return centers_old, id_obj, is_new, lastKey


def extract_roi_from_video(video_path, regions):
    # Callback function for mouse events
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(region_name, img)

    num_rois = len(regions)
    ROIs = []
    print(f'Extracting ROIs from {video_path} with {num_rois} regions of interest')
    # regions = ['gazebo','mcg']
    for i in range(num_rois):
        region_name = regions[i]
        # create frame generator
        video_info = sv.VideoInfo.from_video_path(video_path)
        generator = sv.get_video_frames_generator(video_path)
        # acquire first video frame
        iterator = iter(generator)
        frame = next(iterator)
        # sv.plot_image(frame)

        # Create a window and set the callback function
        img = frame
        cv2.namedWindow(region_name)
        cv2.setMouseCallback(region_name, mouse_callback)

        points = []

        # region_name = input("Enter a name for the region - ")

        while True:
            cv2.imshow(region_name, img)

            # Wait for the user to press any key
            key = cv2.waitKey(1)  # & 0xFF
            if key == 27 or len(points) == 4:  # 'esc' key or 4 points selected
                break

        # Draw lines between the collected points
        if len(points) == 4:
            cv2.line(img, points[0], points[1], (0, 0, 255), 2)
            cv2.line(img, points[1], points[2], (0, 0, 255), 2)
            cv2.line(img, points[2], points[3], (0, 0, 255), 2)
            cv2.line(img, points[3], points[0], (0, 0, 255), 2)
            cv2.imshow(region_name, img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        for i in range(2):
            cv2.waitKey(1)

        # Return the coordinates and plot the frame with counter line
        # sv.plot_image(img)
        print("Selected Points:", points)

        # Extract the rectangular ROI based on the selected points
        roi_x = min(points, key=lambda x: x[0])[0]
        roi_y = min(points, key=lambda x: x[1])[1]
        roi_width = max(points, key=lambda x: x[0])[0] - roi_x
        roi_height = max(points, key=lambda x: x[1])[1] - roi_y

        # Extract ROI from the frame
        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        x_range = [min(coord[0] for coord in points), max(coord[0] for coord in points)]
        y_range = [min(coord[1] for coord in points), max(coord[1] for coord in points)]

        # Adjust the range based on video width and height
        x_range_final = [max(x_range[0], 0), min(x_range[1], video_info.width - 1)]
        y_range_final = [max(y_range[0], 0), min(y_range[1], video_info.height - 1)]

        rectangle_range = [x_range_final, y_range_final]

        region = {"name": region_name,
                  "polygon": points,
                  "range": rectangle_range
                  }
        ROIs.append(region)

    return ROIs


def detect_pedestrains(video_path, target_dir, regions):
    ### Configurations #Verbose during prediction
    verbose = False
    # Scaling percentage of original frame
    scale_percent = 100
    # model confidence level
    conf_level = 0.25
    # Threshold of centers ( old\new)
    thr_centers = 30
    # Number of max frames to consider a object lost
    frame_max = 10
    # Number of max tracked centers stored
    patience = 100
    # ROI area color transparency
    alpha = 0.3
    # ------------------------------------------------------- # Reading video with cv2
    video = cv2.VideoCapture(video_path)

    # Objects to detect Yolo
    class_IDS = [0]
    # Auxiliary variables
    centers_old = {}

    obj_id = 0
    end = []
    frames_list = []
    count_p = 0
    lastKey = ''
    print(f'[INFO] - Verbose during Prediction: {verbose}')

    # Original information of video
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    print('[INFO] - Original Dim: ', (width, height, fps))

    # Scaling Video for better performance
    print(scale_percent)
    if scale_percent != 100:
        print('[INFO] - Scaling change may cause errors in pixels lines ')
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        print('[INFO] - Dim Scaled: ', (width, height))
    print(scale_percent)

    if '/' in video_path:
        video_name = video_path.split("/")[-1].split(".")[0]
    else:
        video_name = video_path.split(".")[0]

    result_video_name = video_name + ".mp4"
    result_directory = target_dir
    # result_directory = "/Users/kysgattu/FIS/ObjectDetection/Data/ExampleResults"
    # output_path = result_directory + "/" + result_video_name
    annotated_video = result_directory + "/Annotated_" + result_video_name
    VIDEO_CODEC = "MP4V"

    output_video = cv2.VideoWriter(annotated_video,
                                   cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                                   fps, (width, height))
    model = YOLO('yolov8x.pt')
    dict_classes = model.model.names
    rois = extract_roi_from_video(video_path=video_path, regions=regions)
    roi_counts = {roi['name']: 0 for roi in rois}
    count_p_roi = 0
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):  # Outer loop iterating through each frame
        # print(i)
        # _, frame = video.read()

        isFrame, frame = video.read()

        if not isFrame:
            break

        for roi in rois:  # Inner loop iterating through each region of interest

            area_roi = [np.array(roi['polygon'], dtype=np.int32)]

            x_range, y_range = roi['range']
            ROI = frame[y_range[0]:y_range[1], x_range[0]:x_range[1]]

            if verbose:
                print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

            y_hat = model.predict(ROI, conf=conf_level, classes=class_IDS, device='cpu', verbose=False)

            boxes = y_hat[0].boxes.xyxy.cpu().numpy()
            conf = y_hat[0].boxes.conf.cpu().numpy()
            classes = y_hat[0].boxes.cls.cpu().numpy()

            positions_frame = pd.DataFrame(np.concatenate([boxes, conf.reshape(-1, 1), classes.reshape(-1, 1)], axis=1),
                                           columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

            labels = [dict_classes[i] for i in classes]

            for ix, row in enumerate(positions_frame.iterrows()):
                xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')
                center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

                centers_old, id_obj, is_new, lastKey = update_tracking(centers_old, (center_x, center_y), thr_centers,
                                                                       lastKey,
                                                                       i, frame_max)
                roi_counts[roi['name']] += is_new

                cv2.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                for center_x, center_y in centers_old[id_obj].values():
                    cv2.circle(ROI, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.putText(img=ROI, text=id_obj + ':' + str(np.round(conf[ix], 2)),
                            org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),
                            thickness=1)

            # Update count for the current ROI in the dictionary
            # roi_counts[roi['name']] = count_p_roi
            y_coordinate = 40
            for region, person_count in roi_counts.items():
                cv2.putText(img=frame, text=f'Counts People in ROI {region}:{person_count}',
                            org=(30, y_coordinate), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=1)
                y_coordinate += 50

            centers_old = filter_tracks(centers_old, patience)
            # if verbose:
            #     print(counter_in, counter_out)

            overlay = frame.copy()
            cv2.polylines(overlay, pts=area_roi, isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.fillPoly(overlay, area_roi, (255, 0, 0))
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            frames_list.append(frame)
            output_video.write(frame)

    output_video.release()

    # if os.path.exists(annotated_video):
    #     os.remove(annotated_video)

    for region, person_count in roi_counts.items():
        print(f"Number of persons moving {region} is {person_count}")
    print(f"Annotated video saved at {annotated_video}")
    return roi_counts

def browse_video_path():
    file_path = filedialog.askopenfilename(title="Select Video File")
    entry_video_path.delete(0, tk.END)
    entry_video_path.insert(0, file_path)


def browse_target_dir():
    dir_path = filedialog.askdirectory(title="Select Target Directory")
    entry_target_dir.delete(0, tk.END)
    entry_target_dir.insert(0, dir_path)


def clear_values():
    entry_video_path.delete(0, tk.END)
    entry_target_dir.delete(0, tk.END)
    entry_num_regions.delete(0, tk.END)
    entry_region_names.delete(0, tk.END)
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)  # Clear previous results
    result_text.config(state=tk.DISABLED)


def start_detection():
    video_path = entry_video_path.get()
    target_dir = entry_target_dir.get()
    region_names = entry_region_names.get()

    # Validate inputs
    num_regions = entry_num_regions.get()
    if not video_path or not target_dir or not region_names or not num_regions.isdigit():
        messagebox.showerror("Error", "Please provide valid inputs.")
        return
    else:
        num_regions = int(num_regions)
        regions.extend(region_names.split(","))
        if num_regions != len(regions):
            messagebox.showerror("Error", "Number of regions entered does not match the specified number.")
            return
        else:
            try:
                result_text.config(state=tk.NORMAL)
                result_text.delete(1.0, tk.END)  # Clear previous results
                result_text.insert(tk.END, "Detection in progress...\n")
                result_text.config(state=tk.DISABLED)

                final_result = detect_pedestrains(video_path, target_dir, regions)

                result_text.config(state=tk.NORMAL)
                result_text.insert(tk.END, "Detection completed. \nResults:\n")
                for region, count in final_result.items():
                    result_text.insert(tk.END, f"Number of Pedestrains moving {region}: {count}\n")
                result_text.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during detection: {str(e)}")


if __name__ == "__main__":
    # roi_counts = detect_pedestrains(video_path='/Users/kysgattu/FIS/ObjectDetection/Data/MCG_171023_10sec.avi',
    #                             target_dir='/Users/kysgattu/FIS/ObjectDetection/Data/ExampleResults',
    #                             regions=["Towards Gazebo","Along McGauvran"])

    # Create the main window
    root = tk.Tk()
    root.title("Pedestrian Detection App")

    # Create and place widgets
    label_video_path = tk.Label(root, text="Video Path:")
    label_video_path.grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)

    entry_video_path = tk.Entry(root, width=50)
    entry_video_path.grid(row=0, column=1, padx=10, pady=10, columnspan=2)

    button_browse_video = tk.Button(root, text="Browse", command=browse_video_path)
    button_browse_video.grid(row=0, column=3, padx=10, pady=10)

    label_target_dir = tk.Label(root, text="Target Directory:")
    label_target_dir.grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)

    entry_target_dir = tk.Entry(root, width=50)
    entry_target_dir.grid(row=1, column=1, padx=10, pady=10, columnspan=2)

    button_browse_target = tk.Button(root, text="Browse", command=browse_target_dir)
    button_browse_target.grid(row=1, column=3, padx=10, pady=10)

    label_num_regions = tk.Label(root, text="Number of Regions:")
    label_num_regions.grid(row=2, column=0, padx=10, pady=10, sticky=tk.E)

    entry_num_regions = tk.Entry(root, width=50)
    entry_num_regions.grid(row=2, column=1, padx=10, pady=10)

    label_region_names = tk.Label(root, text="Region Names (comma-separated):")
    label_region_names.grid(row=3, column=0, padx=10, pady=10, sticky=tk.E)

    entry_region_names = tk.Entry(root, width=50)
    entry_region_names.grid(row=3, column=1, padx=10, pady=10, columnspan=2)

    button_start_detection = tk.Button(root, text="Start Detection", command=start_detection)
    button_start_detection.grid(row=5, column=0, columnspan=4, pady=10)

    regions = []

    result_text = tk.Text(root, height=10, width=60, state=tk.DISABLED, relief="groove",
                          wrap=tk.WORD)  # ,borderwidth=3)
    result_text.grid(row=6, column=0, columnspan=4, pady=10)

    # Start the Tkinter event loop
    root.mainloop()


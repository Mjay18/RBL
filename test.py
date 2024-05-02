import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os.path
import matplotlib.pyplot as plt

# Initialize variables for video capture
cap = None
frame_count = 0
label = None  # Define label variable globally
root = None   # Define root variable globally
label_text = None  # Define label_text variable globally
selected_gender = None  # Store the gender selected by the user
data = []  # Store data for test.csv

# Function to detect keypoints using ORB algorithm
def detect_keypoints(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints
    keypoints, _ = orb.detectAndCompute(gray, None)
    
    # Convert keypoints to list of (x, y) coordinates
    keypoints_coords = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    
    return keypoints_coords

# Function to extract gait features from keypoints
def extract_gait_features(keypoints):
    # Calculate different gait characteristics
    
    # Step length: Euclidean distance between consecutive keypoints along the walking direction
    step_lengths = []
    for i in range(len(keypoints) - 1):
        x1, y1 = keypoints[i]
        x2, y2 = keypoints[i+1]
        step_lengths.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    
    # Step width: Average distance between left and right keypoints
    left_keypoints = [keypoint for i, keypoint in enumerate(keypoints) if i % 2 == 0]
    right_keypoints = [keypoint for i, keypoint in enumerate(keypoints) if i % 2 != 0]
    step_widths = [abs(left[0] - right[0]) for left, right in zip(left_keypoints, right_keypoints)]
    
    # Step frequency: Number of steps per unit time (e.g., per second)
    step_frequency = len(step_lengths) / video_fps if video_fps else 0
    
    # Stance time: Average time between consecutive left or right keypoints
    stance_times = [1 / video_fps]  # Placeholder value for the first stance time
    for i in range(1, len(keypoints)):
        if i % 2 == 0:  # Left keypoints
            x1, _ = keypoints[i-1]
            x2, _ = keypoints[i]
            stance_times.append(abs(x2 - x1) / video_fps if video_fps else 0)
    
    # Calculate mean of each gait characteristic
    mean_step_length = np.mean(step_lengths) if step_lengths else 0
    mean_step_width = np.mean(step_widths) if step_widths else 0
    mean_stance_time = np.mean(stance_times) if stance_times else 0
    mean_step_frequency = step_frequency
    max_step_length = np.max(step_lengths) if step_lengths else 0
    min_step_length = np.min(step_lengths) if step_lengths else 0
    max_step_width = np.max(step_widths) if step_widths else 0
    min_step_width = np.min(step_widths) if step_widths else 0
    
    return mean_step_length, mean_step_width, mean_stance_time, mean_step_frequency, max_step_length, min_step_length, max_step_width, min_step_width

# Function to handle button click
def select_gender(gender):
    global selected_gender
    selected_gender = gender
    
    # Start displaying frames
    process_frame()

# Function to process each frame for feature extraction
def process_frame():
    global frame_count
    global cap
    global label
    global label_text
    global selected_gender
    global data
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "End of video reached.")
        root.destroy()
        return
    
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert frame to PIL format
    pil_image = Image.fromarray(frame_rgb)
    
    # Convert PIL image to Tkinter-compatible format
    tk_image = ImageTk.PhotoImage(image=pil_image)
    
    # Update label with the new frame
    label.configure(image=tk_image)
    label.image = tk_image
    
    # Detect keypoints in frame
    keypoints = detect_keypoints(frame)
    
    # Extract gait features from keypoints
    gait_features = extract_gait_features(keypoints)
    
    # Append gender and gait features to data list
    data.append([frame_count, selected_gender, *gait_features])
    
    # Increment frame count
    frame_count += 1
    
    # Update label text
    label_text.set(f"Frame {frame_count + 1}: Gender - {selected_gender}")

    # After 100ms (0.1 second interval), call process_frame again
    root.after(100, process_frame)

# Function to calculate gait characteristics from video frames
def calculate_gait_characteristics(video_path):
    global cap
    global frame_count
    global data
    global video_fps
    global label
    global root
    global label_text
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize data list
    data = []
    
    # Initialize frame count
    frame_count = 0
    
    # Initialize Tkinter window for GUI
    root = tk.Tk()
    root.title("Gender Selection")
    
    # Create label for displaying frames
    label_text = tk.StringVar()
    label_text.set("Frame 1: Select Gender")
    label = tk.Label(root, textvariable=label_text)
    label.pack()
    
    # Create button to start processing frames
    start_button = tk.Button(root, text="Start", command=process_frame)
    start_button.pack()
    
    # Start GUI loop
    root.mainloop()
    
    # Close video capture
    cap.release()
    
    # Convert data list to DataFrame
    columns = ['Frame', 'Gender', 'MeanStepLength', 'MeanStepWidth', 'MeanStanceTime', 'MeanStepFrequency', 'MaxStepLength', 'MinStepLength', 'MaxStepWidth', 'MinStepWidth']
    df = pd.DataFrame(data, columns=columns)
    
    # Write DataFrame to test.csv (create a new file each time)
    df.to_csv('test.csv', index=False)

# Function to predict gender using cosine similarity
def predict_gender():
    # Load training data from gait_characteristics.csv
    training_data = pd.read_csv('gait_characteristics.csv')
    
    # Load test data from test.csv
    test_data = pd.read_csv('test.csv')
    
    # Calculate cosine similarity between test data and training data
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(test_data.iloc[:, 2:], training_data.iloc[:, 2:])
    
    # Find the index of the most similar training sample for each test sample
    most_similar_indices = np.argmax(similarities, axis=1)
    
    # Extract predicted genders
    predicted_genders = training_data.iloc[most_similar_indices]['Gender'].values
    
    # Calculate accuracy
    accuracy = np.mean(predicted_genders == test_data['Gender'].values)
    
    # Display accuracy
    #print("Accuracy:", accuracy)
    
    # Plot percentage bar graph
    male_percentage = np.mean(predicted_genders == 'Male') * 100
    female_percentage = 100 - male_percentage
    genders = ['Male', 'Female']
    percentages = [male_percentage, female_percentage]
    
    plt.bar(genders, percentages, color=['blue', 'pink'])
    plt.xlabel('Gender')
    plt.ylabel('Percentage')
    plt.title('Gender Prediction')
    plt.show()

# Example usage
video_path = r"C:\Users\dell\Desktop\rbl\New folder\data\man4.mp4"
calculate_gait_characteristics(video_path)
predict_gender()

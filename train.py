import random
import shutil
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import Model
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam
from ultralytics import YOLO

keypoints_folder = "keypoint"
video_folder = "video"

# Create subfolders for training, validating, and testing
train_folder = os.path.join(keypoints_folder, "train")
val_folder = os.path.join(keypoints_folder, "val")
test_folder = os.path.join(keypoints_folder, "test")

# Generate the keypoints files
yolo = YOLO("yolov8m-pose.pt")
fps = 10
MAX_SEQ = 30
NUM_FEATURES = 8 * 2
EPOCHS = 100
BATCH = 32
LEARNING_RATE = 0.005


def get_keypoints_from_videos():
    # List all files with .avi extension in the dataset
    video_Files = [file for file in os.listdir(video_folder) if file.endswith(".avi")]
    longest_frame = 0
    count = 0
    total_files = len(video_Files)
    for file in video_Files:
        count += 1
        video_path = os.path.join(video_folder, file)
        output_path = os.path.join(keypoints_folder, file).replace(".avi", ".npy")
        print(
            "\rStart to processing file: %s. (%d/%d)"
            % (video_path, count, total_files),
            end=" ",
        )
        cap = cv2.VideoCapture(video_path)

        # check whether the video is opened
        assert cap.isOpened()
        t_msec = 0
        frames = []

        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_msec)
            t_msec = t_msec + 1 / fps * 1000
            ret, img = cap.read()  # Read one frame from the webcam.
            if ret:
                result = yolo(
                    source=img,
                    show=False,
                    conf=0.5,
                    save=False,
                    verbose=False,
                    stream=False,
                )[0]

                # Display the annotated frame
                # cv2.imshow("YOLOv8 Inference", annotated_frame)

                result_keypoint = result.keypoints.xyn.cpu().numpy()[0]

                if len(result_keypoint) == 17:
                    # left_shoulder_x, left_shoulder_y = result_keypoint[get_keypoint.LEFT_SHOULDER]
                    # print("left_shoulder:%s, %s" % (left_shoulder_x, left_shoulder_y))
                    # Extracted following 8 keypoints from YOLO pose keypoints
                    # LEFT_SHOULDER:  int = 5
                    # RIGHT_SHOULDER: int = 6
                    # LEFT_ELBOW:     int = 7
                    # RIGHT_ELBOW:    int = 8
                    # LEFT_HIP:       int = 11
                    # RIGHT_HIP:      int = 12
                    # LEFT_KNEE:      int = 13
                    # RIGHT_KNEE:     int = 14
                    extracted_rows = result_keypoint[np.r_[5:9, 11:15], :]
                    frames.append(extracted_rows)
            else:
                break
        reshaped = np.array(frames).reshape(len(frames), 16)
        # record and print the longest frame
        longest_frame = (
            reshaped.shape[0] if reshaped.shape[0] > longest_frame else longest_frame
        )
        np.save(output_path, reshaped)
    print("Longest frame length:", longest_frame)


def create_training_datasets(train_ratio=0.8, val_ratio=0.15):
    # Seperate data into training, validating and testing.
    # Check if the directory already exists
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(val_folder):
        shutil.rmtree(val_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get a list of all the sample data files
    all_files = [file for file in os.listdir(keypoints_folder) if file.endswith(".npy")]

    # Shuffle the list of files
    random.shuffle(all_files)

    # Calculate the number of files for each subset
    num_train = int(len(all_files) * train_ratio)
    num_val = int(len(all_files) * val_ratio)

    # Copy the files to the respective subfolders
    for i, file in enumerate(all_files):
        if i < num_train:
            shutil.copy(
                os.path.join(keypoints_folder, file), os.path.join(train_folder, file)
            )
        elif i < num_train + num_val:
            shutil.copy(
                os.path.join(keypoints_folder, file), os.path.join(val_folder, file)
            )
        else:
            shutil.copy(
                os.path.join(keypoints_folder, file), os.path.join(test_folder, file)
            )


def get_sequence_model():
    # Create an Input layer with the specified input shape
    input_layer = Input(shape=(30, NUM_FEATURES), batch_size=BATCH)

    # Pass the output of the Input layer to the LSTM layer
    lstm_layer1 = LSTM(128, return_sequences=True, dropout=0.02)(input_layer)
    lstm_layer2 = LSTM(64, dropout=0.02)(lstm_layer1)

    # Create the rest of the model using the functional API
    dense_layer1 = Dense(64, activation="tanh")(lstm_layer2)
    dense_layer2 = Dense(64, activation="relu")(dense_layer1)
    dense_layer3 = Dense(64, activation="relu")(dense_layer2)
    output_layer = Dense(2, activation="softmax")(dense_layer3)

    # Create the model using the Model class
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", "recall"],
    )
    return model


def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history["loss"][-1]
    acc = history.history["accuracy"][-1]
    plt.title("Loss: %.3f, Accuracy: %.3f" % (loss, acc))


# Load and prepare data from training
def prepare_all_videos(dir, max_seq, num_features):
    files = os.listdir(dir)
    num_samples = len(files)

    labels = np.zeros([len(files), 3], dtype=int)

    frame_features = np.empty(
        (num_samples, max_seq, num_features)
    )  # Create an empty array to store the stacked matrices

    # one-hot encoding for classes
    for i, filename in enumerate(files):
        filename = os.path.join(dir, filename)
        if "A043" in filename:
            labels[i][0] = 1
        elif "A009" in filename:
            labels[i][1] = 1
        else:
            labels[i][2] = 1

        loaded = np.load(filename)

        # Normalize every odd column by position of Left should x
        # Normalize every even column by position of Left should y
        x = loaded[:, 0].reshape(loaded.shape[0], 1)
        y = loaded[:, 1].reshape(loaded.shape[0], 1)

        # make sure all x >0
        if all(x) > 0:
            loaded[:, ::2] = loaded[:, ::2] / x
        if all(y) > 0:
            loaded[:, 1::2] = loaded[:, 1::2] / y

        # Compare to max_frame drop the exceed frames or pad the frames with zeros
        if loaded.shape[0] > max_seq:
            output_matrix = loaded[:max_seq, :]
        else:
            padding_rows = max_seq - loaded.shape[0]
            zero_padding = np.zeros((padding_rows, 16))
            output_matrix = np.vstack((loaded, zero_padding))

        frame_features[i] = output_matrix

    return frame_features, labels


def train():
    train_data, train_labels = prepare_all_videos(train_folder, MAX_SEQ, NUM_FEATURES)
    val_data, val_labels = prepare_all_videos(val_folder, MAX_SEQ, NUM_FEATURES)

    # Train model
    mymodel = get_sequence_model()
    history = mymodel.fit(
        train_data,
        train_labels,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_data=(val_data, val_labels),
    )
    plot_loss_accuracy(history)

    # Save model
    save_path = "model/falldetection.keras"
    mymodel.save(save_path, overwrite=True)
    print("Model saved to:", save_path)


def help_function():
    print("Usage: python train.py [task]")
    print("Available tasks: ")
    print(
        "keypoint : Preprocess all the avi files in video folder to get keypoints, and split them into train, validation and tests datasets."
    )
    print(
        "train     : train the model using training datasets and save the result in model folder."
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        help_function()
    else:
        function_name = sys.argv[1]
        if function_name == "keypoint":
            get_keypoints_from_videos()
            create_training_datasets()
        elif function_name == "train":
            train()
        else:
            print("You need to input one argument to select which task to run.")
            help_function()

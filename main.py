import cv2
import os
import numpy as np
import keras
from ultralytics import YOLO
from utils import UnscentedKalmanFilterPredictor

# params
FPS = 10
MAX_SEQ = 30
NUM_FEATURES = 8 * 2
yolo = YOLO("yolov8m-pose.pt")
fall_detection_model = keras.models.load_model("model/falldetection.keras")

def normalize_frames_to_input_matrix(frames):
    reshaped = np.array(frames).reshape(len(frames), 16)
    # Normalize every odd column by position of Left should x
    # Normalize every even column by position of Left should y
    x = reshaped[:, 0].reshape(reshaped.shape[0], 1)
    y = reshaped[:, 1].reshape(reshaped.shape[0], 1)

    if all(x) > 0:
        reshaped[:, ::2] = reshaped[:, ::2] / x
    if all(y) > 0:
        reshaped[:, 1::2] = reshaped[:, 1::2] / y

    if reshaped.shape[0] > MAX_SEQ:
        output_matrix = reshaped[:MAX_SEQ, :]
    else:
        padding_rows = MAX_SEQ - reshaped.shape[0]
        zero_padding = np.zeros((padding_rows, 16))
        output_matrix = np.vstack((reshaped, zero_padding))

    output_matrix = output_matrix.reshape(1, MAX_SEQ, NUM_FEATURES)
    return output_matrix


def put_text_on_frame(frame, is_fall):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    text_position = (400, 300)
    font_tickness = 6

    # get an orange color for fall and green for not fall
    color = (0, 0, 255) if is_fall else (0, 255, 0)

    cv2.putText(
        frame,
        "Fall" if is_fall else "Not Fall",
        text_position,
        font,
        font_scale,
        color,
        font_tickness,
    )


def detect_fall_from_video(save=False, input_video=""):
    cap = cv2.VideoCapture(input_video)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    if save:
        out = cv2.VideoWriter("output.avi", fourcc, 5, (1920, 1080))

    ACTIVE = ["Fall", "Other"]

    # check whether the video is opened
    assert cap.isOpened()

    mymodel = fall_detection_model
    t_msec = 0
    frames = []

    # initialize 8 Kalman filter predictors in an array for the 8 keypoints
    kalman_filters = [UnscentedKalmanFilterPredictor(FPS) for _ in range(8)]

    while cv2.waitKey(1) < 0:
        # extract video frame by FPS setting
        cap.set(cv2.CAP_PROP_POS_MSEC, t_msec)
        t_msec = t_msec + 1 / FPS * 1000

        # Read one frame from the input cap.
        ret, img = cap.read()

        if ret:  # Successfully read an image
            yolokeypoints = yolo(
                source=img, show=False, conf=0.5, save=False, verbose=False
            )[0]
            annotated_frame = yolokeypoints.plot(
                kpt_line=True, labels=False, boxes=False, kpt_radius=0
            )
            if save:
                out.write(annotated_frame)

            result_keypoint = yolokeypoints.keypoints.xyn.cpu().numpy()[0]
            if len(result_keypoint) == 17:
                # Extracted following 8 keypoints from YOLO pose keypoints,
                # we only need these 8 points for prediction which can increase the generalization of the model.
                # LEFT_SHOULDER:  int = 5
                # RIGHT_SHOULDER: int = 6
                # LEFT_ELBOW:     int = 7
                # RIGHT_ELBOW:    int = 8
                # LEFT_HIP:       int = 11
                # RIGHT_HIP:      int = 12
                # LEFT_KNEE:      int = 13
                # RIGHT_KNEE:     int = 14
                extracted_rows = result_keypoint[np.r_[5:9, 11:15], :]
                if np.any(extracted_rows) > 0:
                    frames.append(extracted_rows)

                     # update Kalman filter for each keypoint
                    for i, kalman_filter in enumerate(kalman_filters):
                        x, y = extracted_rows[i]
                        kalman_xy = kalman_filter.predict()
                        print(kalman_filter.kalman.x, x, y)
                        if x == 0 and y == 0:
                            print("kalman filter predicting:", kalman_xy)
                            extracted_rows[i] = kalman_xy
                        else:
                            kalman_filter.update(np.array([x, y]))
                            
                    # if frames reach the max sequence, remove the first one.
                    if len(frames) == MAX_SEQ + 1:
                        frames = frames[1:]

                    if len(frames) == MAX_SEQ:
                        matrix = normalize_frames_to_input_matrix(frames)
                        result = mymodel.predict(matrix)
                        category_id = int(np.argmax(result, axis=-1))
                        active = ACTIVE[category_id]
                        print("Predicted action:", active)
                        # display this active on the frame
                        put_text_on_frame(annotated_frame, active == "Fall")

            # Display the annotated frame
            cv2.imshow("Display", annotated_frame)

        else:
            print("End of video")
            matrix = normalize_frames_to_input_matrix(frames)
            result = mymodel.predict(matrix)
            category_id = int(np.argmax(result, axis=-1)[0])
            active = ACTIVE[category_id]
            print("Predicted action:", active)
            # display this active on the frame
            put_text_on_frame(annotated_frame, active == "Fall")
            # Display the annotated frame
            cv2.imshow("Display", annotated_frame)
            # wait any keyinput to continue
            key = cv2.waitKey(0)
            break

    if save:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


def detect_fall_from_camera(save=False):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    if save:
        out = cv2.VideoWriter("output.avi", fourcc, 5, (1920, 1080))
    ACTIVE = ["Fall", "Other"]
    # check whether the video is opened
    assert cap.isOpened()
    mymodel = fall_detection_model
    frames = []
    
    fall_frames = 0

    # initialize 8 Kalman filter predictors in an array for the 8 keypoints
    kalman_filters = [UnscentedKalmanFilterPredictor(FPS) for _ in range(8)]
    while cv2.waitKey(1) < 0:
        # Read one frame from the camera
        ret, img = cap.read()
        if ret:  # Successfully read an image
            yolokeypoints = yolo(
                source=img, show=False, conf=0.5, save=False, verbose=False
            )[0]
            annotated_frame = yolokeypoints.plot(
                kpt_line=True, labels=False, boxes=False, kpt_radius=0
            )
            result_keypoint = yolokeypoints.keypoints.xyn.cpu().numpy()[0]
            if len(result_keypoint) == 17:
                # Extracted following 8 keypoints from YOLO pose keypoints,
                # we only need these 8 points for prediction which can increase the generalization of the model.
                # LEFT_SHOULDER:  int = 5
                # RIGHT_SHOULDER: int = 6
                # LEFT_ELBOW:     int = 7
                # RIGHT_ELBOW:    int = 8
                # LEFT_HIP:       int = 11
                # RIGHT_HIP:      int = 12
                # LEFT_KNEE:      int = 13
                # RIGHT_KNEE:     int = 14
                extracted_rows = result_keypoint[np.r_[5:9, 11:15], :]
                if np.any(extracted_rows) > 0:
                    frames.append(extracted_rows)
                    
                    # update Kalman filter for each keypoint
                    for i, kalman_filter in enumerate(kalman_filters):
                        x, y = extracted_rows[i]
                        kalman_xy = kalman_filter.predict()
                        print(kalman_filter.kalman.x, x, y)
                        if x == 0 and y == 0:
                            print("kalman filter predicting:", kalman_xy)
                            extracted_rows[i] = kalman_xy
                        else:
                            kalman_filter.update(np.array([x, y]))
                            
                    # if frames reach the max sequence, remove the first one.
                    if len(frames) == MAX_SEQ + 1:
                        frames = frames[1:]
                    if len(frames) == MAX_SEQ:
                        matrix = normalize_frames_to_input_matrix(frames)
                        result = mymodel.predict(matrix)
                        category_id = int(np.argmax(result, axis=-1))
                        active = ACTIVE[category_id]
                        print("Predicted action:", active)
                        if active == "Fall":
                            fall_frames += 1
                        else:
                            fall_frames -=1
                        fall_frames = max(0, fall_frames)
                        # if reaches 3 fall frames, we consider it as a fall
                        if fall_frames >= 5:
                            fall_detected = True
                        else:
                            fall_detected = False
                        # display this active on the frame
                        put_text_on_frame(annotated_frame, fall_detected)

            # Display the annotated frame
            cv2.imshow("Display", annotated_frame)
            if save:
                out.write(annotated_frame)
        else:
            print("Failed to read frame from camera")
            break
    if save:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_mode = "camera"

    # set this to true if you want to save the output video    
    save_output_video = True
    
    if run_mode == "camera":
        detect_fall_from_camera(save=save_output_video)
    elif run_mode == "video":
        # list avi files under video folder and call detect_fall function for each video
        for file in os.listdir("test_video"):
            if file.endswith(".avi"):
                detect_fall_from_video(input_video=os.path.join("test_video", file), save=save_output_video)

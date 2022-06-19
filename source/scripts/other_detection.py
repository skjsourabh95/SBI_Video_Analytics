
import torch
import numpy as np
import cv2
from time import time
from imutils.video import FPS
import imutils

class OD:
    def __init__(self, video_path, model_path,yolo_path):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.video_path = video_path
        self.model = self.load_model(model_path,yolo_path)
        self.classes = self.model.names
        print(self.classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        return cv2.VideoCapture(self.video_path)

    def load_model(self, model_path,yolo_path):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        
        labels, cord, conf = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-2], results.xyxyn[0][:, -2]
        return labels, cord , conf

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        if results:
            labels, cord, conf= results
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            for i in range(n):
                row = cord[i]
                if conf[i] >= 0.5:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 5)
                    cv2.putText(frame, f'{self.classes[int(labels[i])]}:{round(float(conf[i]), 2)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        count  = 0
        fps = FPS().start()
        while True:

            ret, frame = cap.read()
            if frame is None:
                break

            if count % 5 == 0:
                results = self.score_frame(frame)
                print(results)
                frame = self.plot_boxes(results, frame)
            count+=1

            cv2.imshow('Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        fps.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = OD(
        video_path = "../sample_data/fire.mp4", #0 - webcam # "../sample_data/fire.mp4" # "../data/weapon.mp4"
        model_path="../models/yolov5/runs/train/fire-theft/weights/best_both.pt",
        yolo_path="../models/yolov5")
    detector()
import cv2
from ultralytics import YOLO
import datetime
import os


def process_frame(frame, model, last_detection_time, cooldown_seconds):
    # Run YOLOv8 inference
    results = model(frame, verbose=False)[0]

    # Check for 1 detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label == '1' and conf > 0.6:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check cooldown
            now = datetime.datetime.now()
            if not last_detection_time or (now - last_detection_time).total_seconds() > cooldown_seconds:
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"Detected: {timestamp}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Save image
                filename = f"detections/bicycle_{timestamp.replace(':', '-')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Bicycle detected and saved: {filename}")
                return frame, now

    return frame, last_detection_time


def main():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Create output directory
    os.makedirs("detections", exist_ok=True)

    # Get user input for analysis type
    print("Select analysis type:")
    print("1. Image file")
    print("2. Video file")
    print("3. Webcam")

    choice = input("Enter your choice (1-3): ")

    cooldown_seconds = 5
    last_detection_time = None

    if choice == '1':
        # Image analysis
        image_path = input("Enter the path to your image file: ")
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not load image")
            return

        processed_frame, _ = process_frame(frame, model, None, cooldown_seconds)
        cv2.imshow("YOLOv8 Bicycle Detection", processed_frame)
        cv2.waitKey(0)

    elif choice == '2':
        # Video file analysis
        video_path = input("Enter the path to your video file: ")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return

    elif choice == '3':
        # Webcam analysis
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

    else:
        print("Invalid choice")
        return

    # Process video (both file and webcam)
    if choice in ['2', '3']:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, last_detection_time = process_frame(frame, model, last_detection_time, cooldown_seconds)
            cv2.imshow("YOLOv8 Bicycle Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

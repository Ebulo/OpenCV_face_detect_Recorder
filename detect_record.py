import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
det_stop_time = None
timer_started = False
Sec_to_rec = 1

frame_size = (int(cap.get(3)), int(cap.get(4)))
vdo_fmt_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# output = cv2.VideoWriter("Video.mp4", vdo_fmt_fourcc, 20, frame_size)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            crr_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            output = cv2.VideoWriter(f"{crr_time}.mp4", vdo_fmt_fourcc, 40, frame_size)
            print("Started Recording!")
    elif detection:
        if timer_started:
            if time.time() - det_stop_time >= Sec_to_rec:
                detection = False
                timer_started = False
                output.release()
                print("Stoped Recording!")
        else:
            timer_started = True
            det_stop_time = time.time()
    
    if detection:
        output.write(frame)

    for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

    # cv2.imshow("Detector", frame)

    if cv2.waitKey(1) == "q":
        break

output.release()
cap.release()
cv2.destroyAllWindows()

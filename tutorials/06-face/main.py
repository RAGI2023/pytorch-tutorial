import cv2
import mediapipe as mp

print("程序启动")

# ✅ 初始化 mediapipe
mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 摄像头打不开")
    exit()

print("✅ 摄像头打开成功")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ 读取失败")
        break

    # 👉 BGR → RGB（必须！）
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 👉 人脸检测
    result = face.process(rgb)

    # 👉 画框
    if result.detections:
        for det in result.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
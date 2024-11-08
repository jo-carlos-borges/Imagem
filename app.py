import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
solucao_reconhecimento_rosto = mp.solutions.face_detection

reconhecer_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils


while True:

    verificador, frame = webcam.read()

    if not verificador:
        break

    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = Hands.process(frameRGB)
    handPoints = results.multi_hand_landmarks
    h, w, _ = frame.shape
    pontos = []

    if handPoints:
        for points in handPoints:
            desenho.draw_landmarks(frame, points, hands.HAND_CONNECTIONS)

            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                pontos.append((cx,cy))

            dedos = [8,12,16,20]
            contador = 0

            if pontos:
                if pontos[4][0] < pontos[3][0]:
                    contador += 1
                for x in dedos:
                    if pontos[x][1] < pontos[x-2][1]:
                        contador += 1

            cv2.rectangle(frame, (80, 10), (200,110), (255, 0, 0), -1)
            cv2.putText(frame,str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)                

    lista_rostos = reconhecer_rostos.process(frame)

    if lista_rostos.detections:

        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    cv2.imshow("Rostos na Webcam", frame)

    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
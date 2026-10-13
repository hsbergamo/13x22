
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
AUDIO_ACERTO = './audio/urna_eletronica.mp3'
AUDIO_ERRO = './audio/buzz.mp3'

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 4.0
FONT_THICKNESS = 8

DEBUG = False


def desenhar_numero(frame, numero, cor=(255, 255, 255)):
    """Desenha o número centralizado em fonte grande."""
    texto = str(numero)
    (tw, th), _ = cv2.getTextSize(texto, FONT, FONT_SCALE, FONT_THICKNESS)
    x = (frame.shape[1] - tw) // 2
    y = (frame.shape[0] + th) // 2
    cv2.putText(frame, texto, (x+180, y), FONT, FONT_SCALE, cor, FONT_THICKNESS, cv2.LINE_AA)


def main():
    pygame.mixer.init()
    acerto_som = pygame.mixer.Sound(AUDIO_ACERTO)
    erro_som = pygame.mixer.Sound(AUDIO_ERRO)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        detector.close()
        return

    start_time = time.time()
    ultima_position = None
    som_atual = None
    position = None

    def tocar_som(qual):
        nonlocal ultima_position, som_atual
        if qual not in (1, 2) or qual == ultima_position:
            return
        if som_atual:
            som_atual.stop()
        som = acerto_som if qual == 1 else erro_som
        som.play()
        som_atual = som
        ultima_position = qual

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # frame[:, :, 1] = (frame[:, :, 1] * 0.1).astype(np.uint8)  # Green
            # frame[:, :, 0] = (frame[:, :, 0] * 0.4).astype(np.uint8)  # Blue
            
            frame[:,:,2] = (frame[:,:,2] * 0.8).astype(np.uint8)  # red
            frame[:,:,1] = (frame[:,:,1] * 0.0).astype(np.uint8)  # Green
            frame[:,:,0] = (frame[:,:,0] * 0.0).astype(np.uint8)  # Blue  

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                index_x = landmarks[8].x * frame.shape[1]
                index_y = landmarks[8].y * frame.shape[0]
                thumb_x = landmarks[4].x * frame.shape[1]
                thumb_y = landmarks[4].y * frame.shape[0]

                if index_y < thumb_y:
                    position = 1
                elif index_y > thumb_y:
                    position = 2
                else:
                    position = None

                if DEBUG:
                    print(f"position: {position} | index_y: {index_y:.1f} | thumb_y: {thumb_y:.1f}")

                tocar_som(position)

                p1 = (int(index_x), int(index_y))
                p2 = (int(thumb_x), int(thumb_y))

                if position == 1:
                    p3 = (int(index_x), int(thumb_y))
                else:
                    p3 = (int(thumb_x), int(index_y))

                cv2.line(frame, p1, p3, (0, 255, 0), 2)
                cv2.line(frame, p3, p2, (0, 255, 0), 2)
                cv2.circle(frame, p1, 5, (0, 255, 0), -1)
                cv2.circle(frame, p2, 5, (0, 255, 0), -1)
            else:
                position = None

            if position == 1:
                desenhar_numero(frame, 13, (255, 255, 255))
            elif position == 2:
                desenhar_numero(frame, 22, (200, 0, 0))
            else:
                desenhar_numero(frame, '?')

            cv2.imshow('13 acertos ou 22 erros', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

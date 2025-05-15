import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# --- Sostituisci con l'ordine ESATTO delle classi usato durante l'addestramento ---
CLASS_NAMES = [
    "0","1", "2", "3", "4", "5", "pistola", "letsgosky","ok"]  # 🚨 Modifica secondo il tuo dataset!

# --- Carica il modello ---
model = tf.keras.models.load_model('Phand3.1.h5')

# --- Funzione di normalizzazione ---
def normalize_hand(hand):
    hand = hand.reshape(-1, 3)
    base = hand[0]  # Polso
    mid = hand[9]  # Base del medio
    distanza = np.linalg.norm(base - mid)
    hand_normalized = (hand - base) / distanza
    return hand_normalized.flatten()

# --- Inizializza MediaPipe ---
mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Rileva landmarks
    results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        raw_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

        # Normalizza
        processed_data = normalize_hand(raw_data).reshape(1, -1)

        # Predici
        prediction = model.predict(processed_data, verbose=0)[0]
        class_id = np.argmax(prediction)
        label = CLASS_NAMES[class_id]  # 🎯 Usa la lista manuale

        # Visualizza
        cv2.putText(image, f"{label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture', image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
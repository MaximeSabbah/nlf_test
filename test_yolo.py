#!/usr/bin/env python3
import cv2
import argparse
from ultralytics import YOLO

def simple_yolo_detection(img_path, model_path, device="cuda:0"):
    # 1. Charger le modèle YOLO
    print(f"Chargement de YOLO ({model_path})...")
    model = YOLO(model_path)

    # 2. Charger l'image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erreur : Impossible de lire l'image {img_path}")
        return

    # 3. Exécuter la détection
    # classes=0 permet de ne détecter que les personnes (person)
    results = model.predict(img, device=device, classes=0, conf=0.25)[0]

    # 4. Parcourir les détections
    print(f"Nombre de personnes détectées : {len(results.boxes)}")
    
    for box in results.boxes:
        # Coordonnées [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Score de confiance
        conf = box.conf[0].item()
        
        # Dessiner le rectangle sur l'image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Ajouter le texte du score
        label = f"Humain: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Sauvegarder et afficher
    output_path = "detection_yolo.jpg"
    cv2.imwrite(output_path, img)
    print(f"Résultat sauvegardé dans : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Chemin de l'image")
    parser.add_argument("--yolo", default="yolo26n.pt", help="Modèle YOLO (.pt)")
    parser.add_argument("--device", default="cuda:0", help="cpu ou cuda:0")
    args = parser.parse_args()

    simple_yolo_detection(args.image, args.yolo, args.device)

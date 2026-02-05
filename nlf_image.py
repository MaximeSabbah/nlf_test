import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
def run_nlf_on_yolo(img_path, yolo_model_path, nlf_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Charger les modèles
    yolo = YOLO(yolo_model_path)
    # On utilise torch.jit.load pour le modèle NLF (TorchScript)
    nlf = torch.jit.load(nlf_model_path).eval().to(device)

    # 2. Préparer les points spécifiques (vos 20 vertices)
    cano_verts = np.load("canonical_verts/smplx.npy")

    indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
               4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,5882,8680,8892,8596,5902,8589,5895
               ,8482,5788,8846,8634]
    
    selected_points = torch.from_numpy(cano_verts[indices]).float().to(device)
    cano_verts_full = np.load("canonical_verts/smplx.npy") # [6890, 3]
    all_points = torch.from_numpy(cano_verts_full).float().to(device)
    # Obtenir les poids pour ces points
    weights = nlf.get_weights_for_canonical_points(all_points)

    # 3. Charger et préparer l'image
    img_bgr = cv2.imread(img_path)
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Tenseur pour NLF : [1, 3, H, W], normalisé 0-1
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().divide(255).unsqueeze(0).to(device)

    # 4. Détection YOLO
    results = yolo.predict(img_bgr, classes=0, conf=0.25, device=device)[0]
    
    # Convertir boxes YOLO [x1, y1, x2, y2] en NLF [x, y, w, h, score]
    nlf_boxes = []
    if len(results.boxes) > 0:
        boxes = results.boxes.xyxy # [N, 4]
        scores = results.boxes.conf.unsqueeze(1) # [N, 1]
        
        # Calcul largeur/hauteur
        wh = boxes[:, 2:] - boxes[:, :2]
        # Concaténer [x1, y1, w, h, score]
        boxes_formatted = torch.cat([boxes[:, :2], wh, scores], dim=1)
        nlf_boxes.append(boxes_formatted)
    else:
        nlf_boxes.append(torch.zeros((0, 5)).to(device))

    # 5. Paramètres Caméra par défaut (Fov 55°)
    # NLF a besoin de matrices pour projeter en 3D
    focal_length = w / (2 * np.tan(np.deg2rad(55) / 2))
    intrinsics = torch.tensor([[
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ]]).float().to(device)
    
    extrinsics = torch.eye(4).unsqueeze(0).to(device) # Caméra à l'origine [1, 4, 4]

    # 6. Inférence NLF
    start_nlf = time.time()
    with torch.inference_mode():
        outputs = nlf.estimate_poses_batched(
            img_tensor,
            nlf_boxes,
            intrinsic_matrix=intrinsics,
            extrinsic_matrix=extrinsics,
            world_up_vector=torch.tensor([0.0, -1.0, 0.0]).to(device),
            weights=weights,
            num_aug=1 # 1 pour la vitesse, 5 pour la précision
        )
    end_nlf = time.time()
    print(f"Temps NLF : {(end_nlf - start_nlf) * 1000:.2f} ms")

    # 7. Exploitation des résultats
    # poses3d est une liste de tenseurs (un par image du batch)
    # Chaque tenseur est de forme [N_personnes, N_points_choisis, 3]
    poses_3d = outputs['poses3d'][0] 
    print(poses_3d)    
    img_resultat = draw_projections(img_bgr, poses_3d, intrinsics)
    cv2.imwrite("resultat_vertices_2d.jpg", img_resultat)

    print(f"Forme de la sortie 3D : {poses_3d.shape}")
    # poses_3d[0] contient les coordonnées (x,y,z) des 20 points pour la 1ère personne
    return poses_3d
def draw_projections(img_bgr, poses_3d, intrinsics):
    # On repasse les données en numpy pour OpenCV
    img_viz = img_bgr.copy()
    poses_3d_np = poses_3d.cpu().numpy()  # Forme [N_personnes, 20, 3]
    K = intrinsics[0].cpu().numpy()       # Forme [3, 3]

    for person_id in range(poses_3d_np.shape[0]):
        # Extraire les 20 points de la personne
        points_3d = poses_3d_np[person_id]

        for pt_3d in points_3d:
            # 1. Projection 3D -> 2D
            # pt_3d est [X, Y, Z]
            x_3d, y_3d, z_3d = pt_3d
            
            # Formule de projection : x = (f*X)/Z + cx
            if z_3d != 0:
                u = (K[0, 0] * x_3d / z_3d) + K[0, 2]
                v = (K[1, 1] * y_3d / z_3d) + K[1, 2]

                # 2. Dessiner le point si il est dans l'image
                cv2.circle(img_viz, (int(u), int(v)), 4, (0, 0, 255), -1) # Point rouge

    return img_viz

# Appel de la fonction
poses = run_nlf_on_yolo("example_image.jpg", "yolo11m.engine", "weights/nlf/nlf_s_multi_0.2.2.torchscript")

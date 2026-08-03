import torch
import numpy as np
import pyvista
from ML.Model import RGCNN_Seg
from ML.Dataset import augment_pointcloud, generate_random_shape, sample_points

def load_model(model_path, vertice=2048, num_classes=2):
    model = RGCNN_Seg(vertice, F=[128,512,1024,512,128,50], K=[6,5,3,1,1,1],
                      num_classes=num_classes, num_categories=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_cloud(model, points, cat=0, prob_threshold=0.8):
    if len(points) > 2048:
        idx = np.random.choice(len(points), 2048, replace=False)
    else:
        idx = np.random.choice(len(points), 2048, replace=True)
    pts = torch.from_numpy(points[idx]).float().unsqueeze(0)
    cat_t = torch.tensor([cat], dtype=torch.long)

    with torch.no_grad():
        logits = model(pts, cat_t)
        probs = torch.softmax(logits, dim=2)
        prob_obj = probs[0, :, 1].cpu().numpy()
        pred = np.zeros(2048, dtype=np.int64)
        pred[prob_obj >= prob_threshold] = 1
    return pred, pts.squeeze(0).cpu().numpy(), prob_obj

if __name__ == "__main__":
    model = load_model("best_model.pth")

    num_points_total = 2048
    num_obj = int(num_points_total * 0.9) # 90% точек принадлежат объекту интереса
    num_bg = num_points_total - num_obj

    #cloud_obj = np.loadtxt("diff_frames/diff_frame_0000.txt")
    cloud_obj = generate_random_shape(num_obj)
    centroid = np.mean(cloud_obj, axis=0)
    cloud_obj = cloud_obj - centroid

    obj_pts = sample_points(cloud_obj, num_obj)
    obj_labels = np.full(num_obj, 1, dtype=np.int64)

    bg_pts = generate_random_shape(num_bg)
    bg_labels = np.full(num_bg, 0, dtype=np.int64)

    all_pts = np.vstack([obj_pts, bg_pts])
    all_labels = np.concatenate([obj_labels, bg_labels])
    shuffle_idx = np.random.permutation(num_points_total)
    all_pts = all_pts[shuffle_idx]
    all_labels = all_labels[shuffle_idx]

    all_pts = augment_pointcloud(all_pts)
    labels, points, probs = predict_cloud(model, all_pts, prob_threshold=0.85)

    if sum(labels==1)>500:
        interest_points = points[labels==1]

        x_max = np.max(interest_points[:, 0])
        x_min = np.min(interest_points[:, 0])
        y_max = np.max(interest_points[:, 1])
        y_min = np.min(interest_points[:, 1])
        z_max = np.max(interest_points[:, 2])
        z_min = np.min(interest_points[:, 2])
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        z_delta = z_max - z_min
        volume = x_delta*y_delta*z_delta
        print("Размеры продукции и их параметры на конвейере:")
        print(f"Ширина изготовленной продукции: {x_delta:.3f} у.е.")
        print(f"Глубина изготовленной продукции: {y_delta:.3f} у.е.")
        print(f"Высота изготовленной продукции: {z_delta:.3f} у.е.")
        print(f"Объём изготовленной продукции: {volume:.3f} кубических у.е.")
    else:
        print("Изготовленной продукции под камерой глубины не наблюдается")

    plotter = pyvista.Plotter()
    plotter.add_points(points, scalars=labels, cmap=['blue', 'red'], point_size=5)
    plotter.show(interactive=True)
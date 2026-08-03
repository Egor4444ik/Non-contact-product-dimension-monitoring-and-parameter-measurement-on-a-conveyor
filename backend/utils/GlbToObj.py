import sys
import trimesh

def glb_to_obj_lines(input_glb: str, output_txt: str) -> None:
    """
    Конвертирует GLB в текстовый файл со строками v, vn, vt, f (как в OBJ).
    Все геометрические данные объединяются в один объект.
    """
    # Загружаем как единый меш (все трансформации применены)
    mesh = trimesh.load(input_glb, force='mesh')

    if mesh is None or len(mesh.vertices) == 0:
        print("Геометрия не найдена.")
        return

    vertices = mesh.vertices                     # позиции (Nx3)
    faces = mesh.faces                           # треугольники (Mx3)
    normals = mesh.vertex_normals                # нормали (вычисляются, если отсутствуют)
    texcoords = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        texcoords = mesh.visual.uv               # текстурные координаты (Nx2)

    # Флаги наличия данных (длины должны совпадать с количеством вершин)
    has_vn = normals is not None and len(normals) == len(vertices)
    has_vt = texcoords is not None and len(texcoords) == len(vertices)

    with open(output_txt, 'w', encoding='utf-8') as f:
        # Вершины
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # Нормали
        if has_vn:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        # Текстурные координаты
        if has_vt:
            for t in texcoords:
                f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        # Полигоны (треугольники)
        for face in faces:
            i1, i2, i3 = face + 1               # индексы с 1
            if has_vn and has_vt:
                f.write(f"f {i1}/{i1}/{i1} {i2}/{i2}/{i2} {i3}/{i3}/{i3}\n")
            elif has_vn and not has_vt:
                f.write(f"f {i1}//{i1} {i2}//{i2} {i3}//{i3}\n")
            elif not has_vn and has_vt:
                f.write(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")
            else:
                f.write(f"f {i1} {i2} {i3}\n")

    print(f"Сохранено: {len(vertices)} вершин, {len(faces)} граней → {output_txt}")

if __name__ == "__main__":
    input_glb = "teapot.glb"
    output_txt = "teapot.obj"
    glb_to_obj_lines(input_glb, output_txt)
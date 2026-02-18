import uproot
import awkward as ak
import numpy as np


def get_tree_metadata(file_path, tree_name="events"):
    """
    Extrae metadatos detallados de un archivo ROOT para diagnóstico.
    """
    with uproot.open(file_path) as file:
        tree = file[tree_name]

        print(f"Reporte de Inspección: {file_path} ---")
        print(f"Eventos totales: {tree.num_entries}")
        print(f"Tamaño en disco: {file.file.source.num_bytes / 1e6:.2f} MB")

        # Analiza las primeras 10 ramas
        print("\nDetalle de Ramas (Top 10):")
        print(f"{'Branch Name':<25} | {'Dtype':<15} | {'Avg Size/Evt'}")
        print("-" * 60)

        for name in tree.keys()[:10]:
            interpretation = tree[name].interpretation
            # Calcula el tamaño promedio
            size = tree[name].compressed_bytes / tree.num_entries
            print(f"{name:<25} | {str(interpretation):<15} | {size:.2f} bytes")


if __name__ == "__main__":
    path = "../04_Data_Sandbox/HZZ.root"
    get_tree_metadata(path)

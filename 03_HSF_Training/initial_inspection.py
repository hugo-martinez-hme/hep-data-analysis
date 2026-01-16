import uproot

# Abrimos el archivo
file = uproot.open("04_Data_Sandbox/HZZ.root")

# 1. Listar lo que hay dentro del archivo
print("Objetos en el archivo:", file.keys())

# 2. Acceder al árbol de eventos (suele llamarse 'events' o 'events;1')
# Nota: uproot añade ';1' para indicar la versión del objeto
tree = file["events"] 

# 3. Ver las "Ramas" (Branches), que son las columnas de datos
print("\nPrimeras 10 columnas (Branches):")
print(tree.keys()[:10])

# 4. Ver cuántas colisiones (entradas) hay
print(f"\nNúmero total de eventos: {tree.num_entries}")
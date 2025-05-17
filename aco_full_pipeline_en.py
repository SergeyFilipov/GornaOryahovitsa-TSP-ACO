import pandas as pd
import numpy as np
import random
import os
import folium
import openrouteservice
import time

from openrouteservice import distance_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from math import ceil
from folium import Map, Marker, PolyLine, Icon
from matplotlib import pyplot as plt

with open(r"C:\Users\Sergey Filipov\Desktop\Проект Адаптация\Разработки\Basic Key.txt", "r") as f:
    api_key = f.read().strip()

#  Block 1: Visualization of all stops and accessibility
base_dir = os.path.dirname(os.path.abspath(__file__))
stops_file = os.path.join(base_dir, "stops_gorna_oryahovitsa_final.xlsx")
df_all = pd.read_excel(stops_file)
df_all['Спирка №'] = df_all['Спирка №'].astype(str).str.extract(r'(\d+)').astype(int)
df_all[['Latitude', 'Longitude']] = df_all['Latitude, Longitude'].str.split(',', expand=True).astype(float)

# Centre on the map
center_lat_all = df_all['Latitude'].mean()
center_lon_all = df_all['Longitude'].mean()
m_all = folium.Map(location=[center_lat_all, center_lon_all], zoom_start=13)

# Add tags by accessibility
for _, row in df_all.iterrows():
    color = "green" if row['Достъпност'] else "red"
    tooltip = f"{row['Спирка №']} – {row['Име на спирка']}"
    popup = f"""
    <b>Спирка №:</b> {row['Спирка №']}<br>
    <b>Име:</b> {row['Име на спирка']}<br>
    <b>Достъпност:</b> {"Да" if row['Достъпност'] else "Не"}
    """
    folium.Marker(
        location=(row['Latitude'], row['Longitude']),
        tooltip=tooltip,
        popup=popup,
        icon=folium.Icon(color=color)
    ).add_to(m_all)

# Save HTML map
map_all_output = os.path.join(base_dir, "all_stops_map.html")
m_all.save(map_all_output)
print(f"🗺️ The map with all stops is saved in: {map_all_output}")

# Block 2.1: Generating or loading a matrix
distance_matrix_path = os.path.join(base_dir, "distance_matrix_part1.xlsx")
generate_new_matrix = False  # Change to True if you want to generate a new matrix (requires API key and internet)

if generate_new_matrix:
    sources = df_all[['Longitude', 'Latitude']].values.tolist()
    destinations = sources.copy()
    client = openrouteservice.Client(key=api_key)
    response = distance_matrix.distance_matrix(
        client,
        locations=sources + destinations,
        profile='driving-car',
        metrics=['distance'],
        sources=list(range(len(sources))),
        destinations=list(range(len(sources), len(sources) + len(destinations))),
        resolve_locations=True,
        units='m'
    )

    distance_df = pd.DataFrame(response['distances'], index=df_all['Спирка №'], columns=df_all['Спирка №'])
    distance_df.to_excel(distance_matrix_path)
    print("✅ The matrix is generated and saved in:", distance_matrix_path)
else:
    distance_df = pd.read_excel(distance_matrix_path, index_col=0)
    print("✅ An existing matrix with distances of:", distance_matrix_path)
    distance_df.columns = distance_df.columns.astype(str).str.extract(r'(\d+)').astype(int).squeeze()
    distance_df.index = distance_df.index.astype(str).str.extract(r'(\d+)').astype(int).squeeze()

print(f"📏 Matrix with distances: {distance_df.shape[0]} row × {distance_df.shape[1]} columns")

# Block 2.2: Generate or load duration matrix
duration_matrix_path = os.path.join(base_dir, "time_matrix_part1.xlsx")
generate_new_duration_matrix = False  # Change to True if you want to generate a new matrix (requires API and internet)

if generate_new_duration_matrix:
    batch_size = 35
    all_coords = df_all[['Longitude', 'Latitude']].values.tolist()
    n_stops = len(all_coords)
    n_batches = ceil(n_stops / batch_size)
    duration_batch_dir = os.path.join(base_dir, "duration_batches")
    os.makedirs(duration_batch_dir, exist_ok=True)

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_stops)

        source_coords = all_coords[start_idx:end_idx]
        locations = source_coords + all_coords

        sources = list(range(len(source_coords)))
        destinations = list(range(len(source_coords), len(source_coords) + n_stops))

        print(f"▶ Batch processing {i + 1}: СП {df_all['Спирка №'].iloc[start_idx]} \
        до СП {df_all['Спирка №'].iloc[end_idx - 1]}")

        response = distance_matrix.distance_matrix(
            client,
            locations=locations,
            profile='driving-car',
            metrics=['duration'],
            sources=sources,
            destinations=destinations,
            resolve_locations=True,
            units='m'
        )

        df_result = pd.DataFrame(
            response['durations'],
            index=df_all['Спирка №'].iloc[start_idx:end_idx],
            columns=df_all['Спирка №']
        )

        output_file = os.path.join(duration_batch_dir, f"duration_part_{i + 1}.xlsx")
        df_result.to_excel(output_file)
        print(f"✅ Saved file: {output_file}")

    print("🎉 All batches with duration are ready.")
else:
    duration_df = pd.read_excel(duration_matrix_path, index_col=0)
    print(f"✅ Duration matrix is loaded by: {duration_matrix_path}")
    duration_df.columns = duration_df.columns.astype(str).str.extract(r'(\d+)').astype(int).squeeze()
    duration_df.index = duration_df.index.astype(str).str.extract(r'(\d+)').astype(int).squeeze()

print(f"⏱️ Matrix with duration: {duration_df.shape[0]} row × {duration_df.shape[1]} columns")

# === Block 2.3: Stop clustering (by Silhouette Score) ===
coords = df_all[['Latitude', 'Longitude']].to_numpy()
MIN_CLUSTERS = 3
MAX_CLUSTERS = 10
sil_scores = []
k_range = range(MIN_CLUSTERS, MAX_CLUSTERS + 1)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    score = silhouette_score(coords, labels)
    sil_scores.append(score)

optimal_k = k_range[sil_scores.index(max(sil_scores))]
kmeans_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
df_all['cluster'] = kmeans_final.fit_predict(coords)

print(f"📌 Optimal number of clusters according to Silhouette Score: {optimal_k}")
print("\n📋 List of stops by cluster:")
for cl in range(optimal_k):
    cluster_stops = df_all[df_all['cluster'] == cl][['Спирка №', 'Име на спирка']]
    stop_list = [f"СП {row['Спирка №']} – {row['Име на спирка']}" for _, row in cluster_stops.iterrows()]
    print(f"🔸 Cluster {cl + 1} ({len(cluster_stops)} stops): " + ", ".join(stop_list))

# Block 3: ACO Algorithm
print("▶ Launch ACO for distance and time...")
alpha_values = [0.5, 1.0, 1.5]
beta_values = [2.0, 4.0, 6.0]
evaporation_rates = [0.3, 0.5, 0.7]
num_ants_list = [10, 20]
Q_values = [50, 100]
num_iterations = 50


def run_aco(matrix_df):
    dist_matrix = matrix_df.to_numpy()
    stop_names = matrix_df.index.tolist()
    num_stops = len(stop_names)

    def calculate_distance(path):
        total = sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        return total + dist_matrix[path[-1]][path[0]]

    global_best_distance = float("inf")
    global_best_path = []
    global_best_params = {}

    for alpha in alpha_values:
        for beta in beta_values:
            for evap in evaporation_rates:
                for num_ants in num_ants_list:
                    for Q in Q_values:
                        pheromone = np.ones((num_stops, num_stops))
                        best_distance = float("inf")
                        best_path = []

                        for _ in range(num_iterations):
                            all_paths = []
                            all_distances = []

                            for _ in range(num_ants):
                                unvisited = list(range(num_stops))
                                path = [random.choice(unvisited)]
                                unvisited.remove(path[0])

                                while unvisited:
                                    current = path[-1]
                                    probabilities = []
                                    for next_stop in unvisited:
                                        pher = pheromone[current][next_stop] ** alpha
                                        visibility = (1 / dist_matrix[current][next_stop]) ** beta
                                        probabilities.append(pher * visibility)

                                    probabilities = np.array(probabilities)
                                    probabilities /= probabilities.sum()
                                    next_city = random.choices(unvisited, weights=probabilities, k=1)[0]
                                    path.append(next_city)
                                    unvisited.remove(next_city)

                                distance = calculate_distance(path)
                                all_paths.append(path)
                                all_distances.append(distance)

                                if distance < best_distance:
                                    best_distance = distance
                                    best_path = path

                            pheromone *= (1 - evap)
                            for i in range(num_ants):
                                path = all_paths[i]
                                for j in range(len(path) - 1):
                                    a, b = path[j], path[j+1]
                                    pheromone[a][b] += Q / all_distances[i]
                                    pheromone[b][a] += Q / all_distances[i]
                                pheromone[path[-1]][path[0]] += Q / all_distances[i]

                        if best_distance < global_best_distance:
                            global_best_distance = best_distance
                            global_best_path = best_path
                            global_best_params = {
                                "alpha": alpha,
                                "beta": beta,
                                "evaporation": evap,
                                "num_ants": num_ants,
                                "Q": Q
                            }

    best_named = [str(stop_names[i]).replace("СП ", "") for i in global_best_path]
    return best_named, global_best_distance, global_best_params


# Run ACO on both matrices
best_named_distance, best_score_distance, best_params_distance = run_aco(distance_df)
best_named_duration, best_score_duration, best_params_duration = run_aco(duration_df)

print("\n✅ [DISTANCE] Best route:", " → ".join(best_named_distance))
print(f"📏 Total length: {round(best_score_distance / 1000, 2)} км")
print("⚙️ Parameters:", best_params_distance)

print("\n✅ [DURATION] Best route:", " → ".join(best_named_duration))
print(f"⏱️ Total time: {round(best_score_duration / 60, 2)} мин")
print("⚙️ Parameters:", best_params_duration)


# Block 4: Route visualisation
df_all['Достъпност'] = df_all['Достъпност'].astype(str).str.strip().str.lower().map({'да': True, 'не': False})


def visualize_route(best_named_list, df_ref, output_path, color):
    route_coords = []
    m = None

    # Collect the coordinates along the route
    for sp_name in best_named_list:
        row = df_ref[df_ref['Спирка №'] == int(sp_name)]
        if not row.empty:
            lat = row['Latitude'].values[0]
            lon = row['Longitude'].values[0]
            route_coords.append((lat, lon))

    if not route_coords:
        print(f"⚠️ Missed preview for {output_path} (no coordinates).")
        return

    center_lat = np.mean([x[0] for x in route_coords])
    center_lon = np.mean([x[1] for x in route_coords])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # 1️⃣ Route markers (blue/green + tooltip with number)
    for idx, (lat, lon) in enumerate(route_coords):
        sp_id = best_named_list[idx]
        name_row = df_ref[df_ref['Спирка №'] == int(sp_id)]
        if not name_row.empty:
            stop_name = name_row['Име на спирка'].values[0]
            tooltip = f"{idx+1}. СП {sp_id} – {stop_name}"
            Marker(
                location=(lat, lon),
                tooltip=tooltip,
                icon=Icon(color=color, icon="info-sign")
            ).add_to(m)

    # 2️⃣ Accessibility markers (all stops - red/green)
    for _, row in df_ref.iterrows():
        acc_color = "green" if row['Достъпност'] else "red"
        tooltip = f"{row['Спирка №']} – {row['Име на спирка']}"
        popup = f"""
        <b>Спирка №:</b> {row['Спирка №']}<br>
        <b>Име:</b> {row['Име на спирка']}<br>
        <b>Достъпност:</b> {"Да" if row['Достъпност'] else "Не"}
        """
        Marker(
            location=(row['Latitude'], row['Longitude']),
            tooltip=tooltip,
            popup=popup,
            icon=Icon(color=acc_color)
        ).add_to(m)

    PolyLine(route_coords, color=color, weight=3, opacity=0.8).add_to(m)
    m.save(output_path)
    print(f"🗺️ Картата е записана в: {output_path}")


map_distance_path = os.path.join(base_dir, "aco_distance_map.html")
map_duration_path = os.path.join(base_dir, "aco_duration_map.html")
visualize_route(best_named_distance, df_all, map_distance_path, "blue")
visualize_route(best_named_duration, df_all, map_duration_path, "green")


# === Block 5: ACO
df_all['Достъпност'] = df_all['Достъпност'].astype(str).str.strip().str.lower().map({'да': True, 'не': False})
center_lat = df_all['Latitude'].mean()
center_lon = df_all['Longitude'].mean()
map_dist = folium.Map(location=[center_lat, center_lon], zoom_start=13)
map_dur = folium.Map(location=[center_lat, center_lon], zoom_start=13)
colormap = plt.get_cmap('tab10')

# 1️⃣ Add accessibility markers (all stops, red/green)
for _, row in df_all.iterrows():
    color = "green" if row['Достъпност'] else "red"
    tooltip = f"{row['Спирка №']} – {row['Име на спирка']}"
    popup = f"""
    <b>Спирка №:</b> {row['Спирка №']}<br>
    <b>Име:</b> {row['Име на спирка']}<br>
    <b>Достъпност:</b> {"Да" if row['Достъпност'] else "Не"}
    """
    for m in [map_dist, map_dur]:
        Marker(
            location=(row['Latitude'], row['Longitude']),
            tooltip=tooltip,
            popup=popup,
            icon=Icon(color=color, icon="ok-sign")
        ).add_to(m)

# 2️⃣ Routes by cluster (distance and time)
for cl in range(df_all['cluster'].nunique()):
    group = df_all[df_all['cluster'] == cl]
    stop_ids = group['Спирка №'].astype(int).tolist()
    distance_sub = distance_df.loc[stop_ids, stop_ids]
    duration_sub = duration_df.loc[stop_ids, stop_ids]
    best_named_dist, _, _ = run_aco(distance_sub)
    best_named_dur, _, _ = run_aco(duration_sub)

    color_rgba = colormap(cl)
    color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in color_rgba[:3])

    # --- DISTANCE ---
    coords_dist = []
    for i, sp in enumerate(best_named_dist):
        row = df_all[df_all['Спирка №'] == int(sp)]
        if not row.empty:
            lat, lon = row['Latitude'].values[0], row['Longitude'].values[0]
            coords_dist.append((lat, lon))
            name = row['Име на спирка'].values[0]
            tooltip = f"{i + 1}. СП {sp} – {name}"
            Marker(
                location=(lat, lon),
                tooltip=tooltip,
                icon=Icon(color="blue", icon="info-sign")
            ).add_to(map_dist)
    if coords_dist:
        PolyLine(coords_dist, color=color_hex, weight=4, opacity=0.8).add_to(map_dist)

    # --- DURATION ---
    coords_dur = []
    for i, sp in enumerate(best_named_dur):
        row = df_all[df_all['Спирка №'] == int(sp)]
        if not row.empty:
            lat, lon = row['Latitude'].values[0], row['Longitude'].values[0]
            coords_dur.append((lat, lon))
            name = row['Име на спирка'].values[0]
            tooltip = f"{i + 1}. СП {sp} – {name}"
            Marker(
                location=(lat, lon),
                tooltip=tooltip,
                icon=Icon(color="green", icon="info-sign")
            ).add_to(map_dur)
    if coords_dur:
        PolyLine(coords_dur, color=color_hex, weight=4, opacity=0.8).add_to(map_dur)

# 3️⃣ Record of the merged cards
map_dist.save(os.path.join(base_dir, "combined_routes_distance.html"))
map_dur.save(os.path.join(base_dir, "combined_routes_duration.html"))
print("🗺️ United route + accessibility maps are saved.")


#  Basic Greedy algorithm for TSP
def greedy_tsp(matrix):
    n = len(matrix)
    unvisited = set(range(1, n))
    path = [0]
    total_cost = 0

    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda city: matrix[current][city])
        total_cost += matrix[current][next_city]
        path.append(next_city)
        current = next_city
        unvisited.remove(next_city)

    total_cost += matrix[current][0]  # return to home town
    path.append(0)
    return path, total_cost


# Implementation of Greedy algorithm  with the distance matrix
start_time = time.time()
greedy_path, greedy_distance = greedy_tsp(distance_df.to_numpy())
greedy_time = time.time() - start_time

print("\n--- Greedy algorithm (distance) ---")
print("Route:", " → ".join(str(distance_df.index[i]) for i in greedy_path))
print(f"📏 Total length: {greedy_distance:.2f} м")
print(f"⏱️ Execution time: {greedy_time:.4f} сек")

# Implementation of Greedy algorithm with the duration matrix
start_time = time.time()
greedy_path_time, greedy_duration = greedy_tsp(duration_df.to_numpy())
greedy_time_duration = time.time() - start_time

print("\n--- Greedy algorithm (duration) ---")
print("Route:", " → ".join(str(duration_df.index[i]) for i in greedy_path_time))
print(f"⏱️ Total time: {greedy_duration:.2f} сек")
print(f"⏱️ Execution time: {greedy_time_duration:.4f} сек")

# Map for Greedy route by distance
greedy_map_dist_path = os.path.join(base_dir, "greedy_distance_map.html")
visualize_route(
    best_named_list=[str(distance_df.index[i]) for i in greedy_path],
    df_ref=df_all,
    output_path=greedy_map_dist_path,
    color="purple"
)

# Map for Greedy route by duration
greedy_map_dur_path = os.path.join(base_dir, "greedy_duration_map.html")
visualize_route(
    best_named_list=[str(duration_df.index[i]) for i in greedy_path_time],
    df_ref=df_all,
    output_path=greedy_map_dur_path,
    color="orange"
)

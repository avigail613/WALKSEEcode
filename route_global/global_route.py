import requests
import os
import sys
from math import radians, cos, sin, sqrt, atan2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OSRM_URL, OSRM_TIMEOUT, ARRIVED_THRESHOLD

class GlobalRoute:
    def __init__(self):
        self.waypoints = []
    # שלב 1: שלח לשרת OSRM — קבל נקודות
    def calculate_route(self, start_gps: tuple, end_gps: tuple) -> list:
        lat1, lon1 = start_gps   
        lat2, lon2 = end_gps   
        url = OSRM_URL.format(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
        try:
            resp = requests.get(
                url,
                params={"overview": "full", "geometries": "geojson"},
                timeout=OSRM_TIMEOUT # לא מחכים יותר מ-OSRM_TIMEOUT שניות
            )
            resp.raise_for_status() # זריקת שגיאה אם HTTP לא 200
        except requests.RequestException as e:
            print(f"[Route] שגיאת OSRM: {e}")
            return []
        data = resp.json()
        if data.get("code") != "Ok": # OSRM החזיר שגיאה
            print(f"[Route] OSRM החזיר שגיאה: {data.get('message', 'unknown')}")
            return []
        coords = data["routes"][0]["geometry"]["coordinates"]
        self.waypoints = [(lat, lon) for lon, lat in coords] # שמירת הנקודות
        print(f"[Route] מסלול התקבל: {len(self.waypoints)} נקודות")
        return list(self.waypoints)

    
    def update_position(self, current_gps: tuple) -> tuple | None:#בדיקה אם הגענו לנקודה הבאה
        if not self.waypoints:
            return None
        dist = _haversine(current_gps, self.waypoints[0])
        if dist < ARRIVED_THRESHOLD:   # הגענו!
            self.waypoints.pop(0)                                 
            print(f"[Route] הגענו לנקודה. נותרו: {len(self.waypoints)}")
        return self.waypoints[0] if self.waypoints else None     

    def get_next(self) -> tuple | None:#מחזיר את הנקודה הבאה
        return self.waypoints[0] if self.waypoints else None

    def has_route(self) -> bool:
        return len(self.waypoints) > 0

    def remaining(self) -> int:
        return len(self.waypoints)

    def clear(self):
        self.waypoints = []

# פונקציה עזרה: מרחק במטרים בין שתי נקודות GPS
def _haversine(p1: tuple, p2: tuple) -> float:
    R = 6_371_000.0
    lat1, lon1 = radians(p1[0]), radians(p1[1])
    lat2, lon2 = radians(p2[0]), radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

from config import MAP_CACHE_PATH

import osmnx as ox

def download(place: str):
    print(f"[download_map] מוריד גרף הליכה עבור: {place}")
    print("[download_map] זה עשוי לקחת כמה דקות תלוי בגודל האזור...")

    G = ox.graph_from_place(place, network_type="walk")

    node_count = len(G.nodes)
    edge_count = len(G.edges)
    print(f"[download_map] הורד בהצלחה: {node_count} צמתים, {edge_count} קשתות")

    os.makedirs(os.path.dirname(MAP_CACHE_PATH), exist_ok=True)

    ox.save_graphml(G, MAP_CACHE_PATH)
    size_mb = os.path.getsize(MAP_CACHE_PATH) / (1024 * 1024)
    print(f"[download_map] נשמר: {MAP_CACHE_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    place = sys.argv[1] if len(sys.argv) > 1 else "Jerusalem, Israel"
    download(place)

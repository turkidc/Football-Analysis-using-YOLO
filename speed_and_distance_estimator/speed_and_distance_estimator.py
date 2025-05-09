import cv2
import numpy as np
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.pixels_per_meter = 10  # Pretpostavljena vrijednost, treba kalibrirati
        self.fps = 30  # Pretpostavljena vrijednost, treba postaviti iz videa
    
    def add_speed_and_distance_to_tracks(self, tracks):
        """Dodaje brzinu i udaljenost u tragove."""
        try:
            # Obradi igrače
            for team in ['team1', 'team2']:
                for player_id, player_data in tracks['players'].items():
                    if 'positions' in player_data and len(player_data['positions']) > 1:
                        # Izračunaj udaljenost
                        total_distance = 0
                        speeds = []
                        
                        for i in range(1, len(player_data['positions'])):
                            prev_pos = player_data['positions'][i-1]
                            curr_pos = player_data['positions'][i]
                            
                            # Izračunaj udaljenost između pozicija
                            distance = np.sqrt(
                                (curr_pos[0] - prev_pos[0])**2 + 
                                (curr_pos[1] - prev_pos[1])**2
                            )
                            
                            # Pretvori u metre
                            distance_meters = distance / self.pixels_per_meter
                            total_distance += distance_meters
                            
                            # Izračunaj brzinu (km/h)
                            speed_kmh = (distance_meters * self.fps * 3.6)  # 3.6 za konverziju u km/h
                            speeds.append(speed_kmh)
                        
                        # Ažuriraj statistiku
                        player_data['total_distance'] = total_distance
                        player_data['speeds'] = speeds
                        player_data['average_speed'] = np.mean(speeds) if speeds else 0
                        player_data['max_speed'] = np.max(speeds) if speeds else 0
            
            return True
            
        except Exception as e:
            print(f"Greška pri računanju brzine i udaljenosti: {str(e)}")
            return False
    
    def draw_speed_and_distance(self, frame, tracks):
        """Crtanje brzine i udaljenosti na frame."""
        try:
            for team in ['team1', 'team2']:
                for player_id, player_data in tracks['players'].items():
                    if 'positions' in player_data and len(player_data['positions']) > 0:
                        # Dohvati zadnju poziciju
                        pos = player_data['positions'][-1]
                        
                        # Pripremi tekst
                        speed_text = f"Speed: {player_data.get('average_speed', 0):.1f} km/h"
                        distance_text = f"Distance: {player_data.get('total_distance', 0):.1f} m"
                        
                        # Nacrtaj tekst
                        cv2.putText(frame, speed_text, (int(pos[0]), int(pos[1] - 20)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, distance_text, (int(pos[0]), int(pos[1])),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Greška pri crtanju brzine i udaljenosti: {str(e)}")
            return frame
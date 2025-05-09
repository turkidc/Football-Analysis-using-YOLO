import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Postavi backend prije importa pyplot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import supervision as sv
from flask import Flask, request, jsonify, send_from_directory, render_template, send_file
import mimetypes
import threading
import time

# Učitaj konfiguraciju
import config

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

# Postavke za OpenCV
cv2.setNumThreads(0)

# Postavke za PyTorch
torch.set_num_threads(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath('.')  # Postavi apsolutnu putanju
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Registriraj MIME tipove
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('application/json', '.json')
mimetypes.add_type('image/png', '.png')

# Globalna varijabla za praćenje statusa obrade
processing_status = {
    'is_processing': False,
    'progress': 0,
    'error': None
}

class FootballAnalyzer:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(video_path)
        
        # Dohvati dimenzije video snimke
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Inicijaliziraj komponente
        self.model = YOLO('yolov5n.pt')
        self.tracker = sv.ByteTrack()  # Koristi supervision ByteTrack za igrače
        self.ball_tracker = sv.ByteTrack()  # Koristi supervision ByteTrack za loptu
        
        # Učitaj prvi frame za inicijalizaciju camera_movement_estimator
        ret, first_frame = self.cap.read()
        if not ret:
            raise Exception("Ne mogu učitati prvi frame iz videa")
        self.camera_movement_estimator = CameraMovementEstimator(first_frame)
        
        self.view_transformer = ViewTransformer(field_size=(width, height))
        self.team_assigner = TeamAssigner()
        self.speed_and_distance_estimator = SpeedAndDistance_Estimator()
        self.player_ball_assigner = PlayerBallAssigner()
        self.stats = {
            'possession': {'team1': 0, 'team2': 0},
            'touches': {'team1': 0, 'team2': 0},
            'players': {
                'team1': {},  # Dictionary za svakog igrača u timu 1
                'team2': {}   # Dictionary za svakog igrača u timu 2
            },
            'ball_possession': []
        }
        self.frame_count = 0
        self.fps = 0
        self.field_size = (0, 0)
        self.frames = []
        
    def update_stats(self, detections, frame):
        """Ažurira statistiku na temelju detekcija."""
        self.frame_count += 1
        
        if self.field_size == (0, 0):
            self.field_size = (frame.shape[1], frame.shape[0])
        
        # Spremi frame za kasniju analizu
        self.frames.append(frame.copy())
        
        # Konvertiraj detekcije u supervision format
        detections_sv = sv.Detections.from_ultralytics(detections[0])
        
        # Treniraj K-means model ako već nije treniran
        if not self.team_assigner.is_fitted:
            self.team_assigner.assign_team_color(frame, detections_sv)
        
        # Prati igrače koristeći supervision ByteTrack
        tracks = self.tracker.update_with_detections(detections_sv)
        
        # Pronađi loptu
        ball_detected = False
        ball_pos = None
        ball_tracks = None
        
        for det in detections_sv:
            if det[3] == 32:  # sports ball class
                ball_detected = True
                try:
                    # Konvertiraj detekciju u supervision format
                    ball_det = sv.Detections(
                        xyxy=det[0].reshape(1, 4),
                        confidence=det[2].reshape(1),
                        class_id=det[3].reshape(1)
                    )
                    
                    # Prati loptu
                    ball_tracks = self.ball_tracker.update_with_detections(ball_det)
                    
                    if len(ball_tracks) > 0:
                        # Izračunaj centar lopte
                        bbox = ball_tracks[0][0]  # Prvi track, xyxy koordinate
                        if len(bbox) == 4:  # Provjeri je li bbox ispravan
                            ball_pos = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                            print(f"Ball position: {ball_pos}")
                except Exception as e:
                    print(f"Greška pri praćenju lopte: {str(e)}")
                break
        
        # Ažuriraj statistiku za svakog igrača
        for track in tracks:
            if track[3] == 0:  # person class
                bbox = track[0]
                track_id = track[4]
                
                # Odredi tim igrača
                team_id = self.team_assigner.get_player_team(frame, bbox, track_id)
                team = 'team1' if team_id == 1 else 'team2'
                
                # Inicijaliziraj igrača ako ne postoji
                if track_id not in self.stats['players'][team]:
                    self.stats['players'][team][track_id] = {
                        'touches': 0,
                        'possession': 0,
                        'positions': [],
                        'speeds': [],
                        'distance': 0
                    }
                
                # Ažuriraj poziciju
                center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                self.stats['players'][team][track_id]['positions'].append(center)
                
                # Izračunaj brzinu i udaljenost
                if len(self.stats['players'][team][track_id]['positions']) > 1:
                    prev_pos = self.stats['players'][team][track_id]['positions'][-2]
                    speed = np.sqrt((center[0] - prev_pos[0])**2 + (center[1] - prev_pos[1])**2)
                    self.stats['players'][team][track_id]['speeds'].append(speed)
                    self.stats['players'][team][track_id]['distance'] += speed
                
                # Ažuriraj dodire i posjed
                if ball_detected and ball_pos is not None:
                    try:
                        dist = np.sqrt((center[0] - ball_pos[0])**2 + (center[1] - ball_pos[1])**2)
                        
                        if dist < 50:  # Prag za dodir
                            self.stats['players'][team][track_id]['touches'] += 1
                            self.stats['touches'][team] += 1
                            print(f"Touch detected: Player {track_id}, Team {team}, Distance {dist:.2f}")
                        
                        if dist < 100:  # Prag za posjed
                            self.stats['players'][team][track_id]['possession'] += 1
                            self.stats['possession'][team] += 1
                            self.stats['ball_possession'].append(team)
                            print(f"Possession detected: Player {track_id}, Team {team}, Distance {dist:.2f}")
                    except Exception as e:
                        print(f"Greška pri računanju udaljenosti: {str(e)}")
                        print(f"Center: {center}, Ball pos: {ball_pos}")
        
        # Očisti statistiku svakih 30 frejmova
        if self.frame_count % 30 == 0:
            self.cleanup_stats()
    
    def cleanup_stats(self):
        """Očisti i ažurira statistiku."""
        for team in ['team1', 'team2']:
            for player_id, stats in self.stats['players'][team].items():
                if stats['speeds']:
                    stats['average_speed'] = np.mean(stats['speeds'])
                else:
                    stats['average_speed'] = 0
                
                # Očisti stare pozicije i brzine
                if len(stats['positions']) > 100:
                    stats['positions'] = stats['positions'][-100:]
                if len(stats['speeds']) > 100:
                    stats['speeds'] = stats['speeds'][-100:]
    
    def generate_player_heatmap(self, player_id):
        """Generira heatmap za pojedinog igrača."""
        try:
            # Prikupi pozicije igrača iz oba tima
            positions = []
            
            # Provjeri tim 1
            if player_id in self.stats['players']['team1']:
                positions.extend(self.stats['players']['team1'][player_id]['positions'])
            
            # Provjeri tim 2
            if player_id in self.stats['players']['team2']:
                positions.extend(self.stats['players']['team2'][player_id]['positions'])
            
            # Ako nema pozicija, vrati None
            if not positions:
                print(f"Nema pozicija za igrača {player_id}")
                return None
            
            # Konvertiraj u numpy array
            positions = np.array(positions)
            
            # Kreiraj prazan heatmap
            heatmap = np.zeros(self.field_size[::-1])
            
            # Dodaj pozicije u heatmap
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < self.field_size[0] and 0 <= y < self.field_size[1]:
                    heatmap[y, x] += 1
            
            # Primijeni Gaussian blur
            heatmap = gaussian_filter(heatmap, sigma=20)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Učitaj sliku igrališta
            field_img = cv2.imread('field.png')
            if field_img is None:
                field_img = np.zeros((self.field_size[1], self.field_size[0], 3), dtype=np.uint8)
                field_img[:] = (0, 100, 0)
            
            field_img = cv2.resize(field_img, (self.field_size[0], self.field_size[1]))
            
            # Konvertiraj heatmap u RGB
            heatmap_rgb = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Kombiniraj s pozadinom
            alpha = 0.7
            combined = cv2.addWeighted(field_img, 1-alpha, heatmap_rgb, alpha, 0)
            
            return combined
            
        except Exception as e:
            print(f"Greška pri generiranju heatmapa za igrača {player_id}: {str(e)}")
            return None
    
    def save_stats(self):
        """Sprema statistiku i vizualizacije."""
        try:
            # Kreiraj direktorij za statistiku ako ne postoji
            stats_dir = os.path.join(self.output_dir, 'stats')
            os.makedirs(stats_dir, exist_ok=True)
            
            # Izračunaj kretanje kamere
            camera_movement = self.camera_movement_estimator.get_camera_movement(self.frames)
            
            # Izračunaj brzinu i udaljenost
            tracks = {
                'players': {
                    'team1': self.stats['players']['team1'],
                    'team2': self.stats['players']['team2']
                }
            }
            
            # Postavi FPS iz videa
            if len(self.frames) > 0:
                self.speed_and_distance_estimator.fps = 30  # Pretpostavljena vrijednost, treba postaviti iz videa
            
            # Izračunaj brzinu i udaljenost
            if not self.speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks):
                print("Upozorenje: Nije moguće izračunati brzinu i udaljenost")
            
            # Konvertiraj numpy tipove u standardne Python tipove
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Konvertiraj statistiku
            stats_to_save = convert_numpy_types(self.stats)
            
            # Spremi statistiku
            stats_file = os.path.join(stats_dir, 'stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats_to_save, f, indent=4)
            
            # Generiraj i spremi heatmap za svakog igrača
            for team in ['team1', 'team2']:
                for player_id in self.stats['players'][team]:
                    heatmap = self.generate_player_heatmap(player_id)
                    if heatmap is not None:
                        heatmap_path = os.path.join(stats_dir, f'heatmap_{player_id}.png')
                        cv2.imwrite(heatmap_path, heatmap)
            
            # Generiraj i spremi graf brzine za svakog igrača
            for team in ['team1', 'team2']:
                for player_id in self.stats['players'][team]:
                    speeds = self.stats['players'][team][player_id].get('speeds', [])
                    if speeds:
                        plt.figure(figsize=(10, 6))
                        plt.plot(speeds)
                        plt.title(f'Brzina - Igrač {player_id}')
                        plt.xlabel('Frame')
                        plt.ylabel('Brzina (km/h)')
                        speed_plot_path = os.path.join(stats_dir, f'speed_{player_id}.png')
                        plt.savefig(speed_plot_path)
                        plt.close()
            
            # Generiraj i spremi graf posjeda
            possession_data = {
                'Tim 1': self.stats['possession']['team1'],
                'Tim 2': self.stats['possession']['team2']
            }
            plt.figure(figsize=(10, 6))
            plt.bar(possession_data.keys(), possession_data.values())
            plt.title('Posjed lopte')
            plt.ylabel('Broj frejmova')
            possession_plot_path = os.path.join(stats_dir, 'possession.png')
            plt.savefig(possession_plot_path)
            plt.close()
            
            print(f"Statistika spremljena u direktorij: {stats_dir}")
            
        except Exception as e:
            print(f"Greška pri spremanju statistike: {str(e)}")
            import traceback
            print("Detalji greške:")
            print(traceback.format_exc())

def process_video(input_path, output_path, status_dict=None):
    """Obrada video snimke."""
    try:
        print(f"Započinjem obradu videa: {input_path}")
        
        # Učitaj YOLO model
        print("Učitavam YOLO model...")
        model = YOLO('yolov5n.pt')
        
        # Inicijaliziraj analizator
        analyzer = FootballAnalyzer(input_path, os.path.dirname(output_path))
        
        # Otvori video
        print("Otvaram video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Ne mogu otvoriti video datoteku: {input_path}")
        
        # Postavi codec i writer
        print("Postavljam video writer...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Kreiraj output direktorij ako ne postoji
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Ukupno frejmova za obradu: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Obradeno frejmova: {frame_count}")
                # Ažuriraj progress
                if status_dict is not None:
                    progress = int((frame_count / total_frames) * 100)
                    status_dict['progress'] = progress
            
            # Obradi frame s YOLO modelom
            results = model(frame, conf=0.3)
            
            # Ažuriraj statistiku
            analyzer.update_stats(results, frame)
            
            # Nacrtaj detekcije
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Nacrtaj bounding box
                    color = (0, 255, 0) if cls == 32 else (255, 0, 0)  # Zelena za loptu, crvena za igrače
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Dodaj label
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Spremi frame
            out.write(frame)
        
        print(f"Obrada završena. Ukupno obradeno frejmova: {frame_count}")
        
        # Spremi statistiku
        analyzer.save_stats()
        
        # Oslobodi resurse
        cap.release()
        out.release()
        
        # Postavi progress na 100%
        if status_dict is not None:
            status_dict['progress'] = 100
            status_dict['is_processing'] = False
        
        return True
        
    except Exception as e:
        print(f"Greška pri obradi videa: {str(e)}")
        import traceback
        print("Detalji greške:")
        print(traceback.format_exc())
        if status_dict is not None:
            status_dict['error'] = str(e)
            status_dict['is_processing'] = False
        return False

def process_video_async(input_path, output_path):
    """Asinkrona obrada videa."""
    global processing_status
    try:
        processing_status['is_processing'] = True
        processing_status['progress'] = 0
        processing_status['error'] = None
        
        # Obradi video
        success = process_video(input_path, output_path, processing_status)
        
        if not success:
            processing_status['error'] = 'Greška pri obradi videa'
            
    except Exception as e:
        processing_status['error'] = str(e)
    finally:
        processing_status['is_processing'] = False

@app.route('/')
def index():
    """Glavna stranica."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Rukovanje uploadom datoteke."""
    global processing_status
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Već se obrađuje video'}), 400
        
    if 'file' not in request.files:
        return jsonify({'error': 'Nema datoteke'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nije odabrana datoteka'}), 400
    
    if file:
        try:
            # Spremi originalnu datoteku
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_videos', 'input_video.mp4')
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            file.save(input_path)
            
            # Kreiraj output direktorij
            output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'output_videos')
            os.makedirs(output_dir, exist_ok=True)
            
            # Obradi video u zasebnoj niti
            output_path = os.path.join(output_dir, 'analyzed_match.mp4')
            thread = threading.Thread(target=process_video_async, args=(input_path, output_path))
            thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Započeta obrada videa',
                'status_url': '/status'
            })
            
        except Exception as e:
            print(f"Greška pri obradi videa: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    """Dohvaća status obrade videa."""
    global processing_status
    return jsonify(processing_status)

@app.route('/check_files')
def check_files():
    """Provjerava postoje li potrebne datoteke."""
    video_path = os.path.join('output_videos', 'analyzed_match.mp4')
    stats_path = os.path.join('output_videos', 'stats', 'stats.json')
    
    return jsonify({
        'video_exists': os.path.exists(video_path),
        'stats_exists': os.path.exists(stats_path)
    })

@app.route('/output_videos/<path:filename>')
def serve_video(filename):
    """Servira video datoteke."""
    try:
        print(f"Pokušavam servirati video: {filename}")
        file_path = os.path.join('output_videos', filename)
        if not os.path.exists(file_path):
            print(f"Datoteka ne postoji: {file_path}")
            return "Datoteka nije pronađena", 404
        return send_file(file_path, mimetype='video/mp4')
    except Exception as e:
        print(f"Greška pri serviranju videa: {str(e)}")
        return str(e), 404

@app.route('/output_videos/stats/<path:filename>')
def serve_stats(filename):
    """Servira statističke datoteke."""
    try:
        print(f"Pokušavam servirati statistiku: {filename}")
        file_path = os.path.join('output_videos', 'stats', filename)
        if not os.path.exists(file_path):
            print(f"Datoteka ne postoji: {file_path}")
            return "Datoteka nije pronađena", 404
            
        if filename.endswith('.json'):
            return send_file(file_path, mimetype='application/json')
        elif filename.endswith('.png'):
            return send_file(file_path, mimetype='image/png')
        else:
            return send_file(file_path)
    except Exception as e:
        print(f"Greška pri serviranju statistike: {str(e)}")
        return str(e), 404

if __name__ == "__main__":
    # Kreiraj potrebne direktorije
    os.makedirs('output_videos', exist_ok=True)
    os.makedirs('output_videos/stats', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)

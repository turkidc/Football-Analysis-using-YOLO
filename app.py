from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify, send_from_directory, Response
import os
import cv2
import tempfile
from main import process_video
import traceback
import json
import mimetypes
import numpy as np

app = Flask(__name__)
app.secret_key = 'football_analysis_secret_key'  # Potrebno za flash poruke

# Onemogući automatsko ponovno učitavanje
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = False

# Dodaj MIME tipove
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('image/png', '.png')

def calculate_team_stats(players):
    """Izračunaj statistiku za tim na temelju statistike igrača."""
    if not players:
        return {'touches': 0, 'possession': 0, 'average_speed': 0, 'distance': 0}
    
    total_touches = sum(player.get('touches', 0) for player in players.values())
    total_possession = sum(player.get('possession', 0) for player in players.values())
    speeds = [player.get('average_speed', 0) for player in players.values() if player.get('average_speed', 0) > 0]
    avg_speed = np.mean(speeds) if speeds else 0
    total_distance = sum(player.get('distance', 0) for player in players.values())
    
    return {
        'touches': total_touches,
        'possession': total_possession,
        'average_speed': avg_speed,
        'distance': total_distance
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('Nema odabrane datoteke', 'error')
        return redirect(url_for('index'))
    
    file = request.files['video']
    if file.filename == '':
        flash('Nema odabrane datoteke', 'error')
        return redirect(url_for('index'))
    
    try:
        # Spremi uploadanu datoteku
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            file.save(tmp_file.name)
            input_path = tmp_file.name
        
        # Obradi video
        output_dir = 'output_videos'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = 'analyzed_' + file.filename
        output_path = os.path.join(output_dir, output_filename)
        
        if process_video(input_path, output_path):
            # Oslobodi privremenu datoteku
            os.unlink(input_path)
            
            # Učitaj statistiku
            stats_file = os.path.join(output_dir, 'stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                # Izračunaj postotke posjeda
                total_frames = stats.get('possession', {}).get('team1', 0) + stats.get('possession', {}).get('team2', 0)
                if total_frames > 0:
                    stats['possession_percentage'] = {
                        'team1': (stats['possession']['team1'] / total_frames) * 100,
                        'team2': (stats['possession']['team2'] / total_frames) * 100
                    }
                else:
                    stats['possession_percentage'] = {'team1': 0, 'team2': 0}
                
                # Izračunaj tim statistiku
                team1_stats = calculate_team_stats(stats.get('players', {}).get('team1', {}))
                team2_stats = calculate_team_stats(stats.get('players', {}).get('team2', {}))
                
                stats['team_stats'] = {
                    'team1': team1_stats,
                    'team2': team2_stats
                }
                
                return render_template('results.html', 
                                     stats=stats,
                                     video_path=output_filename)
            else:
                flash('Statistika nije generirana', 'error')
                return redirect(url_for('index'))
        else:
            flash('Greška pri obradi videa. Provjerite konzolu za detalje.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Greška: {str(e)}\n{error_details}")
        flash(f'Došlo je do greške: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/output_videos/<path:filename>')
def serve_video(filename):
    try:
        return send_from_directory('output_videos', filename, mimetype='video/mp4')
    except Exception as e:
        print(f"Greška pri serviranju videa: {str(e)}")
        return str(e), 404

if __name__ == '__main__':
    # Pokreni Flask bez debug moda
    app.run(debug=False, host='127.0.0.1', port=5000) 
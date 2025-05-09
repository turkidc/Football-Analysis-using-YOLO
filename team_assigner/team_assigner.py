import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = None
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        self.is_fitted = False
        self.min_players_for_training = 4  # Minimalan broj igrača za treniranje
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1).fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self, frame, detections):
        """Određuje boje timova na temelju detekcija igrača."""
        player_colors = []
        
        for det in detections:
            if det[3] == 0:  # person class
                bbox = det[0]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Izreži područje igrača
                player_roi = frame[y1:y2, x1:x2]
                if player_roi.size == 0:
                    continue
                
                # Izračunaj dominantnu boju
                player_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB)
                pixels = player_roi.reshape(-1, 3)
                player_colors.append(np.mean(pixels, axis=0))
        
        # Treniraj model samo ako imamo dovoljno igrača
        if len(player_colors) >= self.min_players_for_training:
            try:
                player_colors = np.array(player_colors)
                self.kmeans.fit(player_colors)
                self.team_colors = self.kmeans.cluster_centers_
                self.is_fitted = True
                print(f"K-means model treniran na {len(player_colors)} igrača")
            except Exception as e:
                print(f"Greška pri treniranju K-means modela: {str(e)}")
                self.is_fitted = False
    
    def get_player_team(self, frame, bbox, track_id):
        """Određuje tim igrača na temelju njegove boje."""
        if not self.is_fitted:
            # Fallback na jednostavnu metodu ako model nije treniran
            return 1 if bbox[0] < frame.shape[1]/2 else 2
        
        try:
            x1, y1, x2, y2 = map(int, bbox)
            player_roi = frame[y1:y2, x1:x2]
            
            if player_roi.size == 0:
                return 1 if bbox[0] < frame.shape[1]/2 else 2
            
            # Izračunaj dominantnu boju igrača
            player_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2RGB)
            pixels = player_roi.reshape(-1, 3)
            player_color = np.mean(pixels, axis=0)
            
            # Predvidi tim
            team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
            return team_id + 1  # Vrati 1 ili 2
            
        except Exception as e:
            print(f"Greška pri određivanju tima: {str(e)}")
            return 1 if bbox[0] < frame.shape[1]/2 else 2

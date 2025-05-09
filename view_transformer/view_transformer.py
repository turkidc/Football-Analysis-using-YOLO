import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self, field_size=(1920, 1080)):
        """
        Inicijalizira ViewTransformer s dimenzijama igrališta.
        
        Args:
            field_size (tuple): Dimenzije video snimke (width, height)
        """
        # Standardne dimenzije nogometnog igrališta u metrima
        self.court_width = 68
        self.court_length = 105
        
        # Postavi početne koordinate za transformaciju
        # Ove koordinate treba prilagoditi vašem videu
        width, height = field_size
        self.pixel_vertices = np.array([
            [width * 0.1, height * 0.9],  # Donji lijevi
            [width * 0.1, height * 0.1],  # Gornji lijevi
            [width * 0.9, height * 0.1],  # Gornji desni
            [width * 0.9, height * 0.9]   # Donji desni
        ])
        
        # Ciljne koordinate u metrima
        self.target_vertices = np.array([
            [0, self.court_width],           # Donji lijevi
            [0, 0],                          # Gornji lijevi
            [self.court_length, 0],          # Gornji desni
            [self.court_length, self.court_width]  # Donji desni
        ])
        
        # Konvertiraj u float32 za OpenCV
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)
        
        # Izračunaj matricu transformacije
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, 
            self.target_vertices
        )
        
        print(f"ViewTransformer inicijaliziran s dimenzijama: {field_size}")
        print(f"Pixel vertices: {self.pixel_vertices}")
        print(f"Target vertices: {self.target_vertices}")

    def transform_point(self, point):
        """
        Transformira točku s video koordinata u stvarne koordinate igrališta.
        
        Args:
            point (tuple): (x, y) koordinate na video snimci
            
        Returns:
            tuple: (x, y) koordinate na igralištu u metrima, ili None ako je točka izvan igrališta
        """
        try:
            if point is None:
                return None
                
            # Provjeri je li točka unutar igrališta
            p = (int(point[0]), int(point[1]))
            is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
            
            if not is_inside:
                print(f"Točka {p} je izvan igrališta")
                return None
            
            # Transformiraj točku
            reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
            transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
            
            return transformed_point.reshape(-1, 2)
            
        except Exception as e:
            print(f"Greška pri transformaciji točke {point}: {str(e)}")
            return None

    def add_transformed_position_to_tracks(self, tracks):
        """
        Dodaje transformirane koordinate u tragove.
        
        Args:
            tracks (dict): Rječnik s tragovima objekata
        """
        try:
            for object_type, object_tracks in tracks.items():
                for frame_num, track in enumerate(object_tracks):
                    for track_id, track_info in track.items():
                        # Dohvati poziciju
                        position = track_info.get('position_adjusted')
                        if position is None:
                            continue
                            
                        # Transformiraj poziciju
                        position = np.array(position)
                        position_transformed = self.transform_point(position)
                        
                        # Spremi transformiranu poziciju
                        if position_transformed is not None:
                            position_transformed = position_transformed.squeeze().tolist()
                        tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
                        
        except Exception as e:
            print(f"Greška pri transformaciji tragova: {str(e)}")
            
    def draw_field_lines(self, frame):
        """
        Crtanje linija igrališta na video snimci.
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            numpy.ndarray: Frame s nacrtanim linijama
        """
        try:
            # Crtaj granice igrališta
            cv2.polylines(frame, [self.pixel_vertices.astype(np.int32)], True, (0, 255, 0), 2)
            
            # Crtaj centar
            center = np.mean(self.pixel_vertices, axis=0).astype(np.int32)
            cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)
            
            return frame
            
        except Exception as e:
            print(f"Greška pri crtanju linija igrališta: {str(e)}")
            return frame
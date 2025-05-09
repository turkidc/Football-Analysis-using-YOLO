import cv2
import numpy as np

def create_field(width=1920, height=1080):
    # Kreiraj zelenu pozadinu
    field = np.zeros((height, width, 3), dtype=np.uint8)
    field[:] = (0, 100, 0)  # Zelena boja za travnjak
    
    # Nacrtaj bijele linije
    white = (255, 255, 255)
    
    # Vanjska linija
    cv2.rectangle(field, (50, 50), (width-50, height-50), white, 2)
    
    # Srednja linija
    cv2.line(field, (width//2, 50), (width//2, height-50), white, 2)
    
    # Srednji krug
    cv2.circle(field, (width//2, height//2), 50, white, 2)
    
    # Kazneni prostori
    # Lijevi
    cv2.rectangle(field, (50, height//2-100), (200, height//2+100), white, 2)
    # Desni
    cv2.rectangle(field, (width-200, height//2-100), (width-50, height//2+100), white, 2)
    
    # Kazneni krugovi
    # Lijevi
    cv2.circle(field, (200, height//2), 50, white, 2)
    # Desni
    cv2.circle(field, (width-200, height//2), 50, white, 2)
    
    # Spremi sliku
    cv2.imwrite('field.png', field)

if __name__ == '__main__':
    create_field() 
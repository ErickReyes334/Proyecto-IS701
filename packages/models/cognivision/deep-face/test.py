from deepface import DeepFace

objs = DeepFace.analyze(
    img_path="img.jpg", ##Inserta la imagen de la misma ruta del archivo para probar
    actions=['emotion']
)

print(objs)
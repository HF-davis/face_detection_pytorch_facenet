# face_detection_pytorch_facenet
facenet image recongnition in pytorch
## Pasos a seguir para utilizar Reconocimiento facial
- Clonar el repositorio

```bash
git clone https://github.com/HF-davis/face_detection_pytorch_facenet.git
```
- Abrir el directorio

```bash
cd face_detection_pytorch_facenet
```

- Creamos un directorio con el nombre train, train contendra los rostros de las personas de las cuales se quiere identificar

```bash
mkdir train
```
- Debemos tener la siguiente estructura para el directorio train, los subdirectorios contendran fotos de las personas que se desea identificar

<img src="./structure.png" alt="directorio train" />

- Instalación de los requerimientos (librerias) que necesita el modelo

```bash
pip install -r requirements.txt
```
- Procedemos a entrenar el modelo, el argumento <strong>-t</strong> sirve para colocar la ruta del directorio train  

```python
python train.py -t './train'
```
- Se creara un archivo (data) <strong>data.pt</strong>, el cual contendra los datos necesarios para que el modelo pueda reconocer a las personas.

## Reconocimiento facial en tiempo real

- El argumento <strong>-s</strong> sirve para indicar que recurso se va a usar, 0 para usar la webcam

```python
python inference.py -s 0
```
## Reconocimiento facial en imagenes

- Para usar RF en imágenes debemos usar el argumento <strong>-i</strong> para indicar la ruta donde se encuentra la imagen que se desea analizar

```python
python inference.py -i './img_path'
```






# Random Image Generator

Este programa genera imágenes aleatorias usando OpenCV y las guarda en formato JPEG. Utiliza un hilo productor y múltiples hilos consumidores para lograr una alta eficiencia.

## Requisitos

- CMake >= 3.10
- OpenCV
- Un compilador compatible con C++17

## Compilación

1. Clona o descarga este repositorio.
2. Abre una terminal en la carpeta del proyecto.
3. Ejecuta los siguientes comandos:

```
mkdir build
cd build
cmake ..
make
```

Esto generará el ejecutable `random_image_generator` dentro del directorio `build`.

## Uso

Ejecuta el programa desde la carpeta `build` con los siguientes parámetros:

```
./random_image_generator -f <fps> -t <duración_en_segundos> -h <hilos_consumidores>
```

### Ejemplo:

```
./random_image_generator -f 50 -t 300 -h 7
```

Este comando generará imágenes durante 5 minutos (300 segundos) a 50 fotogramas por segundo utilizando 7 hilos consumidores para guardar las imágenes.

## Salida

- Las imágenes se guardan en la carpeta `../output/` relativa a la ubicación del ejecutable.
- El programa elimina y vuelve a crear esta carpeta en cada ejecución.
- Los archivos se guardan como img_0.jpg, img_1.jpg, ..., etc.
- El programa imprime estadísticas por segundo (FPS, imágenes encoladas, descartadas).
- Al finalizar, muestra un resumen global con métricas de rendimiento y pérdidas.
  
## Métricas mostradas
- Total de imágenes generadas.
- Imágenes encoladas y guardadas.
- Imágenes descartadas por retraso o cola llena.
- FPS real promedio de guardado.

## Notas

- El sistema descarta imágenes cuando la cola está llena para mantener el ritmo de generación.
- El programa crea y limpia automáticamente la carpeta `output` antes de comenzar.
- Se recomienda ejecutar en sistemas con múltiples núcleos para aprovechar el paralelismo.

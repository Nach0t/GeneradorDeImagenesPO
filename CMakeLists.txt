cmake_minimum_required(VERSION 3.10)
project(RandomImageGenerator)

# Buscar el paquete de OpenCV
find_package(OpenCV REQUIRED)

# Verificar si OpenCV fue encontrado correctamente
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR.")
endif()

# Incluir los directorios de OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Definir el ejecutable y el archivo fuente
add_executable(random_image_generator main.cpp)

# Enlazar las librerías de OpenCV al ejecutable
target_link_libraries(random_image_generator ${OpenCV_LIBS})

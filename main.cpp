/**
 * @file main.cpp
 * @brief Programa multihilo para generar y guardar imágenes aleatorias usando OpenCV.
 * 
 * Este programa lanza un hilo productor que genera imágenes a un FPS configurable,
 * y múltiples hilos consumidores que las guardan como archivos JPEG.
 */

#include <iostream>
#include <deque>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <string>

const int IMAGE_WIDTH = 1920;                  ///< Ancho de las imágenes generadas
const int IMAGE_HEIGHT = 1280;                 ///< Alto de las imágenes generadas
int TARGET_FPS = 50;                           ///< FPS objetivo configurado por el usuario
int DURATION_SECONDS = 300;                    ///< Duración en segundos
int NUM_CONSUMERS = 7;                         ///< Número de hilos consumidores
const std::string outputDir = "../output";  ///< Carpeta de salida para guardar imágenes
const size_t MAX_QUEUE_SIZE = 200;             ///< Tamaño máximo de la cola

namespace fs = std::filesystem;

/**
 * @brief Cola segura para acceso concurrente entre hilos.
 * 
 * Implementa una cola protegida con mutex y condition_variable para sincronización.
 */
template<typename T>
class SafeQueue {
    std::deque<T> queue;           ///< Contenedor subyacente
    std::mutex mtx;                ///< Mutex para sincronización
    std::condition_variable cv;   ///< Variable de condición para bloqueo/espera

public:
    /**
     * @brief Inserta un elemento en la cola.
     * @param value Elemento a insertar.
     */
    void push(T&& value) {
        std::unique_lock<std::mutex> lock(mtx);
        queue.emplace_back(std::move(value));
        cv.notify_one();
    }

    /**
     * @brief Extrae un elemento de la cola de forma segura.
     * @param value Referencia donde se colocará el elemento extraído.
     * @param running Bandera para saber si el productor sigue activo.
     * @return true si se extrajo un elemento, false si se detuvo.
     */
    bool pop(T& value, const std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !queue.empty() || !running.load(); });

        if (queue.empty()) return false;
        value = std::move(queue.front());
        queue.pop_front();
        return true;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    void notify_all() {
        cv.notify_all();
    }
};

// Variables globales
SafeQueue<cv::Mat> imageQueue;
std::atomic<bool> running(true);
std::atomic<int> imagesGenerated(0);
std::atomic<int> imagesSaved(0);
std::atomic<size_t> totalBytesWritten(0);

/**
 * @brief Genera una imagen aleatoria usando valores RGB aleatorios.
 * @return Imagen generada.
 */
cv::Mat generateRandomImage() {
    thread_local cv::Mat img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
    return img.clone();
}

/**
 * @brief Función ejecutada por el hilo productor de imágenes.
 * 
 * Genera imágenes a un ritmo constante definido por TARGET_FPS,
 * y las coloca en una cola compartida.
 */
void imageProducer() {
    using namespace std::chrono;
    auto interval = milliseconds(1000 / TARGET_FPS);
    auto lastPrint = steady_clock::now();
    int fpsCounter = 0;

    while (running.load(std::memory_order_relaxed)) {
        auto start = steady_clock::now();
        auto img = generateRandomImage();

        while (imageQueue.size() >= MAX_QUEUE_SIZE && running.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(milliseconds(1));
        }

        imageQueue.push(std::move(img));
        imagesGenerated.fetch_add(1, std::memory_order_relaxed);
        ++fpsCounter;
        
        auto now = steady_clock::now();
        if (duration_cast<seconds>(now - lastPrint).count() >= 1) {
            std::cout << "[PRODUCER] FPS: " << fpsCounter << "\n";
            fpsCounter = 0;
            lastPrint = now;
        }

        std::this_thread::sleep_until(start + interval);
    }

    imageQueue.notify_all();
}

/**
 * @brief Función ejecutada por los hilos consumidores.
 * 
 * Cada consumidor extrae imágenes de la cola y las guarda en disco en formato JPEG.
 * @param id ID del consumidor.
 */
void imageConsumer(int id) {
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 85 };
    while (true) {
        cv::Mat img;
        if (!imageQueue.pop(img, running)) {
            break;
        }

        int index = imagesSaved.fetch_add(1, std::memory_order_relaxed);
        std::string filename = outputDir + "/img_" + std::to_string(index) + ".jpg";

        std::vector<uchar> buffer;
        if (cv::imencode(".jpg", img, buffer, params)) {
            std::ofstream file(filename, std::ios::binary);
            if (file) {
                file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                totalBytesWritten.fetch_add(buffer.size(), std::memory_order_relaxed);
            }
        }
    }
}

/**
 * @brief Función principal del programa.
 * 
 * Lanza el hilo productor y múltiples hilos consumidores, espera la duración especificada,
 * y luego muestra estadísticas finales del proceso.
 * 
 * @param argc Cantidad de argumentos.
 * @param argv Argumentos de línea de comandos: duración, FPS, consumidores.
 * @return Código de salida del programa.
 */
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Error: Debes ingresar los 3 parámetros requeridos.\n";
        std::cerr << "Uso: " << argv[0] << " <duracion_segundos> <fps> <num_consumidores>\n";
        std::cerr << "Ejemplo: " << argv[0] << " 300 50 7\n";
        return 1;
    }

    int dur = std::atoi(argv[1]);
    int fps = std::atoi(argv[2]);
    int consumers = std::atoi(argv[3]);

    if (dur <= 0 || fps <= 0 || consumers <= 0) {
        std::cerr << "Error: Los valores deben ser enteros positivos.\n";
        return 1;
    }

    DURATION_SECONDS = dur;
    TARGET_FPS = fps;
    NUM_CONSUMERS = consumers;

    std::cout << "Generando imágenes a " << TARGET_FPS
              << " fps durante " << DURATION_SECONDS
              << " segundos usando " << NUM_CONSUMERS << " hilos consumidores...\n";

    if (fs::exists(outputDir)) {
        fs::remove_all(outputDir);
        std::cout << "[INFO] Carpeta de salida eliminada.\n";
    }
    fs::create_directories(outputDir);
    std::cout << "[INFO] Carpeta de salida creada.\n";

    auto startTime = std::chrono::steady_clock::now();

    std::thread producer(imageProducer);
    std::vector<std::thread> consumersThreads;
    for (int i = 0; i < NUM_CONSUMERS; ++i) {
        consumersThreads.emplace_back(imageConsumer, i);
    }

    std::this_thread::sleep_for(std::chrono::seconds(DURATION_SECONDS));
    running.store(false);
    imageQueue.notify_all();

    producer.join();
    for (auto& c : consumersThreads) c.join();

    auto endTime = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    double avgFps = imagesGenerated.load() * 1000.0 / ms;

    std::cout << "----- RESUMEN -----\n";
    std::cout << "Total imágenes generadas: " << imagesGenerated.load() << "\n";
    std::cout << "Total imágenes guardadas: " << imagesSaved.load() << "\n";
    std::cout << "Total datos escritos: " << totalBytesWritten.load() / (1024 * 1024) << " MB\n";
    std::cout << "FPS reales promedio: " << avgFps << "\n";
    std::cout << "----------------\n";

    return 0;
}

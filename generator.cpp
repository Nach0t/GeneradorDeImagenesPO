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

const int IMAGE_WIDTH = 1920;
const int IMAGE_HEIGHT = 1280;
int TARGET_FPS = 50;
int DURATION_SECONDS = 300;
int NUM_CONSUMERS = 7;
const std::string outputDir = "../../output";
const size_t MAX_QUEUE_SIZE = 200;

namespace fs = std::filesystem;

// Cola segura con protección de concurrencia
template<typename T>
class SafeQueue {
    std::deque<T> queue;
    std::mutex mtx;
    std::condition_variable cv;

public:
    void push(T&& value) {
        std::unique_lock<std::mutex> lock(mtx);
        queue.emplace_back(std::move(value));
        cv.notify_one();
    }

    // Nuevo método pop con control de término
    bool pop(T& value, const std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !queue.empty() || !running.load(); });

        if (queue.empty()) return false;  // Salir si no hay más imágenes y se detuvo
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

SafeQueue<cv::Mat> imageQueue;

std::atomic<bool> running(true);
std::atomic<int> imagesGenerated(0);
std::atomic<int> imagesSaved(0);
std::atomic<size_t> totalBytesWritten(0);

// Genera una imagen aleatoria
cv::Mat generateRandomImage() {
    thread_local cv::Mat img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
    return img.clone();  // Clonar para evitar conflictos entre hilos
}

// Productor de imágenes
void imageProducer() {
    using namespace std::chrono;
    auto interval = milliseconds(1000 / TARGET_FPS);
    auto lastPrint = steady_clock::now();
    int fpsCounter = 0;

    while (running.load(std::memory_order_relaxed)) {
        auto start = steady_clock::now();
        auto img = generateRandomImage();

        // Espera si la cola está llena
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

    imageQueue.notify_all();  // Despertar consumidores al terminar
}

// Consumidores que guardan imágenes
void imageConsumer(int id) {
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 85 };
    while (true) {
        cv::Mat img;
        if (!imageQueue.pop(img, running)) {
            break;  // Salir si ya no se generan más imágenes
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

// Función principal
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

    // Preparar carpeta de salida
    if (fs::exists(outputDir)) {
        fs::remove_all(outputDir);
        std::cout << "[INFO] Carpeta de salida eliminada.\n";
    }
    fs::create_directories(outputDir);
    std::cout << "[INFO] Carpeta de salida creada.\n";

    auto startTime = std::chrono::steady_clock::now();

    // Lanzar productor y consumidores
    std::thread producer(imageProducer);
    std::vector<std::thread> consumersThreads;
    for (int i = 0; i < NUM_CONSUMERS; ++i) {
        consumersThreads.emplace_back(imageConsumer, i);
    }

    // Esperar tiempo de duración
    std::this_thread::sleep_for(std::chrono::seconds(DURATION_SECONDS));
    running.store(false);
    imageQueue.notify_all();  // Despertar hilos que podrían estar bloqueados

    // Unir hilos
    producer.join();
    for (auto& c : consumersThreads) c.join();

    auto endTime = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    double avgFps = imagesGenerated.load() * 1000.0 / ms;

    // Resumen final
    std::cout << "----- RESUMEN -----\n";
    std::cout << "Total imágenes generadas: " << imagesGenerated.load() << "\n";
    std::cout << "Total imágenes guardadas: " << imagesSaved.load() << "\n";
    std::cout << "Total datos escritos: " << totalBytesWritten.load() / (1024 * 1024) << " MB\n";
    std::cout << "FPS reales promedio: " << avgFps << "\n";
    std::cout << "----------------\n";

    return 0;
}

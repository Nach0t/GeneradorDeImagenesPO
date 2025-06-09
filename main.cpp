/**
 * @file main.cpp
 * @brief Generador multihilo de imágenes aleatorias y sistema concurrente de guardado en disco.
 *
 * Este programa genera imágenes aleatorias con OpenCV a un FPS definido, las encola
 * y las guarda con múltiples hilos consumidores. Permite observar cuellos de botella
 * y pérdidas en tiempo real, simulando un sistema de procesamiento concurrente.
 */

#include <iostream>
#include <string>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <vector>
#include <atomic>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

/**
 * @struct ImageData
 * @brief Contiene la imagen generada y su índice secuencial.
 */
struct ImageData {
    cv::Mat image; ///< Imagen generada aleatoriamente
    int index;     ///< Índice de la imagen
};

/// Tamaño máximo de la cola de imágenes
const size_t MAX_QUEUE_SIZE = 100;

/// Cola compartida de imágenes
std::deque<ImageData> imageQueue;

/// Mutex para la cola
std::mutex queueMutex;

/// Variable de condición para sincronización
std::condition_variable queueCV;

/// Bandera de finalización de generación
bool finishedGenerating = false;

/// Contadores atómicos para métricas del sistema
std::atomic<int> total_images_generated_count{0};
std::atomic<int> total_images_enqueued_count{0};
std::atomic<int> total_images_dropped_due_to_delay{0};
std::atomic<int> total_images_dropped_due_to_queue{0};
std::atomic<int> total_images_saved_count{0};

/// Tiempo global de finalización para referencia de los consumidores
std::chrono::steady_clock::time_point global_end_time;

/**
 * @brief Genera una imagen aleatoria con colores RGB.
 * @param width Ancho de la imagen.
 * @param height Alto de la imagen.
 * @return Imagen aleatoria de tipo cv::Mat.
 */
cv::Mat generateRandomImage(int width, int height) {
    cv::Mat image(height, width, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    return image;
}

/**
 * @brief Hilo productor que genera imágenes a un FPS determinado y las encola.
 * @param width Ancho de las imágenes.
 * @param height Alto de las imágenes.
 * @param duration_seconds Duración total del experimento.
 * @param fps Cuántas imágenes por segundo generar.
 */
void imageGenerator(int width, int height, int duration_seconds, double fps) {
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);
    global_end_time = end_time;
    std::chrono::duration<double> frame_duration(1.0 / fps);

    int second_counter = 0;
    int gen_this_second = 0;
    int enq_this_second = 0;
    int drop_delay_this_second = 0;
    int drop_queue_this_second = 0;
    int i = 0;
    auto last_print_time = start_time;

    while (std::chrono::steady_clock::now() < end_time) {
        auto current_time = std::chrono::steady_clock::now();
        auto next_frame_time = start_time + frame_duration * (i + 1);

        if (current_time > next_frame_time) {
            total_images_dropped_due_to_delay++;
            drop_delay_this_second++;
            total_images_generated_count++;
            gen_this_second++;
            i++;
            continue;
        }

        std::this_thread::sleep_until(next_frame_time);
        cv::Mat image = generateRandomImage(width, height);

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (imageQueue.size() >= MAX_QUEUE_SIZE) {
                total_images_dropped_due_to_queue++;
                drop_queue_this_second++;
                imageQueue.pop_front();
            }
            imageQueue.push_back({image, i});
            total_images_enqueued_count++;
            enq_this_second++;
        }
        queueCV.notify_all();

        total_images_generated_count++;
        gen_this_second++;
        i++;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count() >= 1) {
            int queue_size;
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                queue_size = imageQueue.size();
            }
            std::cout << "[PRODUCER] Segundo " << second_counter + 1
                      << " | Gen: " << gen_this_second
                      << " | Enq: " << enq_this_second
                      << " | DropDelay: " << drop_delay_this_second
                      << " | DropQueue: " << drop_queue_this_second
                      << " | Cola: " << queue_size << std::endl;
            last_print_time = now;
            second_counter++;
            gen_this_second = enq_this_second = drop_delay_this_second = drop_queue_this_second = 0;
        }
    }
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        finishedGenerating = true;
    }
    queueCV.notify_all();
}

/**
 * @brief Hilo consumidor que guarda las imágenes en el disco.
 * @param output_directory Carpeta de salida.
 * @param extension Extensión de imagen (.jpg, .png).
 * @param saver_id ID del hilo consumidor.
 */
void imageSaver(const std::string& output_directory, const std::string& extension, int saver_id) {
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [] { return !imageQueue.empty() || finishedGenerating; });

        while (!imageQueue.empty()) {
            if (std::chrono::steady_clock::now() >= global_end_time) {
                return;
            }
            ImageData imgData = imageQueue.front();
            imageQueue.pop_front();
            lock.unlock();

            std::string filename = output_directory + "/image_" + std::to_string(imgData.index) + "." + extension;
            if (cv::imwrite(filename, imgData.image)) {
                total_images_saved_count++;
            }

            lock.lock();
        }

        if (finishedGenerating && imageQueue.empty()) {
            break;
        }
    }
}

/**
 * @brief Punto de entrada principal del programa.
 *
 * Lanza el hilo productor y los hilos consumidores según parámetros de línea de comandos.
 *
 * @param argc Cantidad de argumentos.
 * @param argv Argumentos de entrada:
 * -f FPS, -t duración (s), -h hilos consumidores.
 * @return Código de salida.
 */
int main(int argc, char *argv[]) {
    const std::string output_directory = "output";
    if (fs::exists(output_directory)) {
        fs::remove_all(output_directory);
        std::cout << "[INFO] carpeta borrada" << std::endl;
    }
    fs::create_directories(output_directory);
    std::cout << "[INFO] carpeta creada" << std::endl;

    int fps = 50;
    int duration = 10;
    int num_threads = 4;
    std::string extension = "jpg";
    int width = 1920;
    int height = 1280;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) fps = std::stoi(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) duration = std::stoi(argv[++i]);
        else if (arg == "-h" && i + 1 < argc) num_threads = std::stoi(argv[++i]);
    }

    auto start = std::chrono::steady_clock::now();
    std::thread producer(imageGenerator, width, height, duration, fps);
    std::vector<std::thread> consumers;
    for (int i = 0; i < num_threads; ++i) {
        consumers.emplace_back(imageSaver, output_directory, extension, i);
    }

    producer.join();
    for (auto& t : consumers) t.join();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_elapsed = end - start;

    std::cout << "\n--- Resumen Global ---\n";
    std::cout << "Imágenes generadas: " << total_images_generated_count.load() << "\n";
    std::cout << "Imágenes guardadas: " << total_images_saved_count.load() << "\n";
    std::cout << std::fixed << std::setprecision(2)
              << "Tiempo total de ejecución: " << total_elapsed.count() << " segundos\n";
    if (total_elapsed.count() > 0) {
        double overall_fps = total_images_saved_count.load() / total_elapsed.count();
        std::cout << "FPS guardado efectivo: " << overall_fps << "\n";
    }
    std::cout << "Imágenes perdidas por atraso: " << total_images_dropped_due_to_delay.load() << "\n";
    std::cout << "Imágenes perdidas por cola: " << total_images_dropped_due_to_queue.load() << "\n";
    std::cout << "TOTAL imágenes perdidas: "
              << (total_images_dropped_due_to_delay.load() + total_images_dropped_due_to_queue.load()) << "\n";
    return 0;
}

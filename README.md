# Управление БПЛА с помощью VLM Qwen2.5-VL в виртуальном карьере

Симуляционный комплекс для Ubuntu 24.04, ROS 2 Jazzy, Gazebo Sim 8.11.0 (Harmonic), ArduPilot SITL и локальной модели Qwen2.5-VL (`qwen2.5vl:7b` через Ollama).

## Быстрый запуск на компьютере /home/nikita

1. Распакуйте архив `drone_vlm_quarry.zip` в домашний каталог `/home/nikita/`:
   `unzip -o drone_vlm_quarry.zip -d /home/nikita/`

2. Запустите автоматические тесты A-E:
   `cd /home/nikita/drone_vlm_quarry && python3 tests/run_all_tests.py`

3. Откройте мир виртуального карьера в Gazebo Harmonic с GUI:
   `./scripts/run_gazebo.sh`
   (Также работает прямой запуск `gz sim -v4 -r worlds/quarry.sdf`, так как в SDF прописаны полные `file://` URI моделей).

4. Запустите весь комплекс (Gazebo + ArduPilot SITL + мост ROS 2):
   `./scripts/start_all.sh`

5. Откройте русскоязычный терминал оператора:
   `python3 ros2_ws/src/quarry_drone_vlm/quarry_drone_vlm/operator_cli.py`

6. Пример команды оператора:
   `Взлети на высоту 5 метров, найди крупный камень в северной части карьера, пролетай над ним и затем вернись в исходную точку`

7. Остановка всех процессов:
   `./scripts/stop_all.sh`

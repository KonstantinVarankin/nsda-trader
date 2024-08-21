# start_nsda_trader.py

import os
import subprocess
import sys
import time

def check_dependency(command):
    try:
        subprocess.run([command, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"{command} не найден. Пожалуйста, установите {command} и добавьте его в PATH.")
        sys.exit(1)

def run_command(command, cwd=None):
    return subprocess.Popen(command, cwd=cwd, shell=True)

if __name__ == "__main__":
    # Проверяем зависимости
    check_dependency("python")
    check_dependency("npm")

    # Определяем пути к директориям
    project_root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(project_root, "backend")
    frontend_dir = os.path.join(project_root, "frontend")

    # Запуск бэкенда
    print("Запуск бэкенда...")
    backend_process = run_command("python -m uvicorn main:app --reload", cwd=backend_dir)

    # Запуск фронтенда
    print("Запуск фронтенда...")
    frontend_process = run_command("npm start", cwd=frontend_dir)

    # Запуск процесса обучения нейросети
    print("Запуск процесса обучения нейросети...")
    train_process = run_command("python train_model.py", cwd=backend_dir)

    print("Все компоненты NSDA-Trader запущены!")

    try:
        # Держим скрипт запущенным, пока пользователь не прервет его
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nЗавершение работы NSDA-Trader...")
        backend_process.terminate()
        frontend_process.terminate()
        train_process.terminate()
        print("Все процессы завершены.")
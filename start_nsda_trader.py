import os
import subprocess
import sys
import time

def check_and_install_dependencies():
    print("Проверка и установка зависимостей...")
    requirements_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(requirements_file):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
            print("Зависимости успешно установлены.")
        except subprocess.CalledProcessError:
            print("Ошибка при установке зависимостей.")
            sys.exit(1)
    else:
        print("Файл requirements.txt не найден.")

def check_dependency(command):
    try:
        if command == "npm":
            # Проверяем npm с использованием PowerShell
            result = subprocess.run(["powershell", "-Command", "(Get-Command npm -ErrorAction SilentlyContinue) -ne $null"], capture_output=True, text=True)
            return result.returncode == 0
        else:
            subprocess.run([command, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Предупреждение: {command} не найден. Некоторые функции могут быть недоступны.")
        return False

def run_command(command, cwd=None):
    try:
        if isinstance(command, list):
            process = subprocess.Popen(command, cwd=cwd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        else:
            process = subprocess.Popen(command, cwd=cwd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        return process
    except Exception as e:
        print(f"Ошибка при запуске команды '{command}': {e}")
        return None

if __name__ == "__main__":
    check_and_install_dependencies()

    # Проверяем зависимости
    python_available = check_dependency("python")
    npm_available = check_dependency("npm")

    # Определяем пути к директориям
    project_root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(project_root, "backend")
    frontend_dir = os.path.join(project_root, "frontend")

    processes = []

    # Запуск бэкенда
    if python_available:
        print("Запуск бэкенда...")
        backend_process = run_command("python -m uvicorn main:app --reload", cwd=backend_dir)
        if backend_process:
            processes.append(backend_process)
            # Дадим немного времени для запуска и проверим ошибки
            time.sleep(5)
            if backend_process.poll() is not None:
                print("Ошибка при запуске бэкенда:")
                print(backend_process.stderr.read())
            else:
                print("Бэкенд успешно запущен.")
        else:
            print("Не удалось запустить бэкенд.")

    # Запуск фронтенда
    if npm_available:
        print("Запуск фронтенда...")
        frontend_process = run_command(["powershell", "-Command", "cd frontend; npm start"], cwd=project_root)
        if frontend_process:
            processes.append(frontend_process)
            # Дадим немного времени для запуска и проверим ошибки
            time.sleep(5)
            if frontend_process.poll() is not None:
                print("Ошибка при запуске фронтенда:")
                print(frontend_process.stderr.read())
            else:
                print("Фронтенд успешно запущен.")
        else:
            print("Не удалось запустить фронтенд.")
    else:
        print("Невозможно запустить фронтенд без npm.")

    # Запуск процесса обучения нейросети
    if python_available:
        print("Запуск процесса обучения нейросети...")
        train_process = run_command("python train_model.py", cwd=backend_dir)
        if train_process:
            processes.append(train_process)
            print("Процесс обучения нейросети успешно запущен.")
        else:
            print("Не удалось запустить процесс обучения нейросети.")

    if processes:
        print("Запущенные компоненты NSDA-Trader работают.")
        try:
            # Держим скрипт запущенным, пока пользователь не прервет его
            while True:
                for process in processes:
                    output = process.stdout.readline()
                    if output:
                        print(output.strip())
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nЗавершение работы NSDA-Trader...")
            for process in processes:
                process.terminate()
            print("Все процессы завершены.")
    else:
        print("Не удалось запустить ни один компонент NSDA-Trader.")
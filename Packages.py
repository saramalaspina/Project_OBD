import subprocess
import sys
import importlib

def check_pip():
    try:
        import pip
        print("pip is already installed.")
    except ImportError:
        print("pip is not installed. Installation in progress...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        print("pip has been installed successfully.")


def install_package(package_mod, package_nm):
    try:
        importlib.import_module(package_nm)
        print(f"{package_nm} is already installed.")
    except ImportError:
        print(f"{package_nm} is not installed. Installation in progress...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_mod])


def start():
    check_pip()

    packages = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn"
    }

    for package_name, module_name in packages.items():
        install_package(module_name, package_name)

    print("All packages have been installed successfully.\n")

if __name__ == "__main__":
    start()
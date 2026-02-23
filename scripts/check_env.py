"""Preflight environment check for required runtime packages."""

import importlib.util
import platform
import sys

REQUIRED_MODULES = {
    "yaml": "pyyaml",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
}

OPTIONAL_MODULES = {
    "chromadb": "chromadb",
}


def missing_modules():
    missing = []
    for module_name, package_name in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, package_name))
    return missing


def pip_command() -> str:
    if platform.system().lower().startswith("win"):
        return "py -m pip"
    return f"{sys.executable} -m pip"


def main() -> int:
    missing = missing_modules()
    optional_missing = [
        (module_name, package_name)
        for module_name, package_name in OPTIONAL_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if not missing:
        print("Environment check passed: required packages are installed.")
        if optional_missing:
            print("Optional packages not installed (features will gracefully degrade):")
            for module_name, package_name in optional_missing:
                print(f"  - {module_name} (install package: {package_name})")
        return 0

    print("Environment check failed. Missing modules:")
    for module_name, package_name in missing:
        print(f"  - {module_name} (install package: {package_name})")

    pip_cmd = pip_command()
    package_list = " ".join(sorted({pkg for _, pkg in missing}))
    print("\nInstall missing packages with:")
    print(f"  {pip_cmd} install {package_list}")
    print("\nOr install all project deps:")
    print(f"  {pip_cmd} install -r requirements.txt")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

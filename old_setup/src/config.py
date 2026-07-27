from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

CITY_NAME = "Karachi"
KARACHI_LATITUDE = 24.8607
KARACHI_LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"


def create_project_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    create_project_directories()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Forecast city: {CITY_NAME}")
    print("Python environment configured successfully.")
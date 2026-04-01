import sys
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("ERROR: roboflow package not installed. Run: pip install roboflow")
    sys.exit(1)


ROBOFLOW_API_KEY = "FmjeB3R8RvH25mLfBuyN"


def _download_with_fallback(version, preferred="coco-segmentation", fallback="coco"):
    try:
        return version.download(preferred)
    except Exception:
        print(f"  '{preferred}' not available, falling back to '{fallback}' format …")
        return version.download(fallback)


def _get_version(project, preferred_version: int = 1):
    try:
        return project.version(preferred_version)
    except RuntimeError:
        print(f"  Version {preferred_version} not found, scanning for available versions …")
        for v in range(1, 11):
            try:
                ver = project.version(v)
                print(f"  Found version {v}")
                return ver
            except RuntimeError:
                continue
        if hasattr(project, "versions") and callable(project.versions):
            versions = project.versions()
            if versions:
                ver_id = versions[0]["id"].split("/")[-1]
                print(f"  Using version {ver_id} from project metadata")
                return project.version(int(ver_id))
        raise RuntimeError(f"No versions found for project '{project.id}'")


def download_datasets(api_key: str = ROBOFLOW_API_KEY) -> None:
    rf = Roboflow(api_key=api_key)
    print("\n[1/2] Downloading Drywall Taping Area dataset …")
    try:
        project1 = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
        version1 = _get_version(project1, preferred_version=1)
        dataset1 = _download_with_fallback(version1)
        print(f"  → Saved to: {dataset1.location}")
    except Exception as e:
        print(f"  ERROR downloading taping dataset: {e}")
        raise
    print("\n[2/2] Downloading Cracks dataset …")
    try:
        project2 = rf.workspace("university-bswxt").project("crack-bphdr")
        version2 = _get_version(project2, preferred_version=1)
        dataset2 = _download_with_fallback(version2)
        print(f"  → Saved to: {dataset2.location}")
    except Exception as e:
        print(f"  ERROR downloading cracks dataset: {e}")
        raise

    print("\n✓ Both datasets downloaded successfully.")


if __name__ == "__main__":
    download_datasets()

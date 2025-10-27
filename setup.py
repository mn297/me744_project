#!/usr/bin/env python3
"""
Cross-platform setup script for downloading and extracting datasets.
Supports multiple datasets with automatic format detection and extraction.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
import subprocess
from pathlib import Path

# Dataset configuration - add new datasets here
DATASETS = {
    "pointvessel_data.zip": {
        "url": "https://nextcloud.in.tum.de/index.php/s/7ooyYxoP6HyPXQK/download?path=/&files=pointvessel_data.zip&downloadStartSecret=i46y2qmnsg",
        "extract_dir": "pointvessel_data",
    },
    "Fuji-SfM_dataset.rar": {
        "url": "https://zenodo.org/records/3712808/files/Fuji-SfM_dataset.rar?download=1",
        "extract_dir": "fuji_sfm_data",
    },
}


def download_file(url, filename):
    """Download a file from URL with progress indication."""
    print(f"Downloading {filename}...")

    # Try faster downloaders first (aria2, axel, wget with multiple connections)

    # 1. Try aria2 (available on both Linux and Mac via Homebrew)
    if shutil.which("aria2c"):
        print("Using aria2 for faster download...")
        try:
            result = subprocess.run(
                [
                    "aria2c",
                    "--continue=true",  # Resume partial downloads
                    "--max-connection-per-server=8",  # Multiple connections
                    "--split=8",  # Split into 8 segments
                    "--max-concurrent-downloads=1",
                    "--summary-interval=1",  # Progress updates
                    "--retry-wait=3",  # Wait 3 seconds between retries
                    "--max-tries=5",  # Retry up to 5 times
                    "--timeout=30",  # 30 second timeout
                    "--out=" + filename,
                    url,
                ],
                check=True,
            )
            print(f"Download completed: {filename}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"aria2 download failed: {e}")
            print("Trying next downloader...")
        except Exception as e:
            print(f"aria2 error: {e}")
            print("Trying next downloader...")

    # 2. Try axel (good for Mac, available via Homebrew)
    if shutil.which("axel"):
        print("Using axel for faster download...")
        try:
            result = subprocess.run(
                [
                    "axel",
                    "-n",
                    "8",  # 8 connections
                    "-a",  # More concise progress indicator
                    "--output=" + filename,
                    url,
                ],
                check=True,
            )
            print(f"Download completed: {filename}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"axel download failed: {e}")
            print("Trying next downloader...")
        except Exception as e:
            print(f"axel error: {e}")
            print("Trying next downloader...")

    # 3. Try wget with multiple connections (if available)
    if shutil.which("wget"):
        print("Using wget for download...")
        try:
            result = subprocess.run(
                [
                    "wget",
                    "--continue",  # Resume partial downloads
                    "--timeout=30",
                    "--tries=5",
                    "--progress=bar:force",
                    "--output-document=" + filename,
                    url,
                ],
                check=True,
            )
            print(f"Download completed: {filename}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"wget download failed: {e}")
            print("Falling back to urllib...")
        except Exception as e:
            print(f"wget error: {e}")
            print("Falling back to urllib...")

    # Fallback to urllib if no other downloaders work
    try:

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\nDownload completed: {filename}")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False


def extract_zip(filename, extract_dir):
    """Extract a ZIP file."""
    try:
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted {filename} to {extract_dir}/")
        return True
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        return False


def extract_rar(filename, extract_dir):
    """Extract a RAR file using various unrar tools."""
    try:
        # Create extraction directory if it doesn't exist
        Path(extract_dir).mkdir(exist_ok=True)

        # Try different unrar commands in order of preference
        rar_extractors = [
            # Standard unrar tools
            {"cmd": "unrar", "args": ["x", filename, extract_dir + "/"]},
            {"cmd": "rar", "args": ["x", filename, extract_dir + "/"]},
            # The Unarchiver command line tool (popular on Mac)
            {"cmd": "unar", "args": ["-o", extract_dir, filename]},
            # dtrx (universal extractor, if available)
            {"cmd": "dtrx", "args": ["-n", "-d", extract_dir, filename]},
        ]

        for extractor in rar_extractors:
            cmd = extractor["cmd"]
            args = extractor["args"]

            if shutil.which(cmd):
                print(f"Trying {cmd} for RAR extraction...")
                result = subprocess.run(
                    [cmd] + args,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print(f"Extracted {filename} to {extract_dir}/ using {cmd}")
                    return True
                else:
                    print(f"{cmd} failed: {result.stderr}")

        # If no command-line tools found, try Python's rarfile library
        try:
            import rarfile

            with rarfile.RarFile(filename, "r") as rar_ref:
                rar_ref.extractall(extract_dir)
            print(f"Extracted {filename} to {extract_dir}/ using Python rarfile")
            return True
        except ImportError:
            pass

        # If all else fails, provide helpful installation instructions
        print(f"Warning: Cannot extract {filename} - no RAR extraction tool found.")
        print("Please install a RAR extractor:")

        if sys.platform.startswith("darwin"):  # macOS
            print("  Mac options (choose one):")
            print("    brew install unar          # The Unarchiver CLI (recommended)")
            print("    brew install unrar         # Traditional unrar")
            print("    brew install dtrx          # Universal extractor")
        elif sys.platform.startswith("linux"):
            print("  Linux options:")
            print("    sudo apt install unrar     # Ubuntu/Debian")
            print("    sudo yum install unrar     # RHEL/CentOS")
            print("    sudo apt install unar      # The Unarchiver CLI")

        print("  Or install Python rarfile: pip install rarfile")
        return False

    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        return False


def clone_and_install_sam2():
    """Clone SAM2 repository and install it in editable mode."""
    # sam2_dir = Path.home() / "sam2"
    sam2_dir = Path("site-packages/sam2")
    sam2_url = "https://github.com/facebookresearch/segment-anything-2.git"

    print("\n--- Setting up SAM2 ---")

    # Check if git is available
    if not shutil.which("git"):
        print("✗ git is not installed. Please install git first.")
        return False

    # Clone repository if it doesn't exist
    if sam2_dir.exists():
        print(f"✓ {sam2_dir} already exists, skipping clone")
        # Check if it's a git repository
        if (sam2_dir / ".git").exists():
            print("✓ Valid git repository found")
        else:
            print(f"⚠ {sam2_dir} exists but is not a git repository")
            return False
    else:
        print(f"Cloning SAM2 to {sam2_dir}...")
        try:
            result = subprocess.run(
                ["git", "clone", sam2_url, str(sam2_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✓ Successfully cloned SAM2 to {sam2_dir}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone SAM2: {e.stderr}")
            return False
        except Exception as e:
            print(f"✗ Error cloning SAM2: {e}")
            return False

    # Install in editable mode
    print(f"Installing SAM2 in editable mode...")

    # Detect if we're using uv or traditional pip
    install_commands = []

    # Try uv first (if available)
    if shutil.which("uv"):
        print("✓ Using uv for installation")
        install_commands.append(["uv", "pip", "install", "-e", str(sam2_dir)])

    # Fallback to traditional pip
    install_commands.append(
        [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)]
    )

    # Try each installation method
    for cmd in install_commands:
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ Successfully installed SAM2 in editable mode")
            return True
        except subprocess.CalledProcessError as e:
            if cmd == install_commands[-1]:  # Last command failed
                print(f"✗ Failed to install SAM2: {e.stderr}")
                return False
            # Try next method
            continue
        except Exception as e:
            if cmd == install_commands[-1]:  # Last command failed
                print(f"✗ Error installing SAM2: {e}")
                return False
            # Try next method
            continue

    return False


def is_directory_empty(directory_path):
    """Check if a directory exists and is empty."""
    if not Path(directory_path).exists():
        return True  # If it doesn't exist, consider it "empty"

    if not Path(directory_path).is_dir():
        return False  # If it's not a directory, it's not empty

    # Check if directory has any files or subdirectories
    try:
        return not any(Path(directory_path).iterdir())
    except PermissionError:
        print(f"Warning: Permission denied accessing {directory_path}")
        return False


def extract_file(filename, extract_dir):
    """Extract a file based on its extension."""
    file_ext = Path(filename).suffix.lower()

    if file_ext == ".zip":
        return extract_zip(filename, extract_dir)
    elif file_ext == ".rar":
        return extract_rar(filename, extract_dir)
    else:
        print(f"Unsupported archive format: {file_ext}")
        return False


def main():
    """Main setup function."""
    print("=== Cross-Platform Dataset Setup ===")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")

    # Check for fast downloaders availability
    fast_downloaders = []

    if shutil.which("aria2c"):
        fast_downloaders.append("aria2")
        print("✓ aria2 detected - using fast multi-connection downloads")

    if shutil.which("axel"):
        fast_downloaders.append("axel")
        print("✓ axel detected - using fast multi-connection downloads")

    if shutil.which("wget"):
        fast_downloaders.append("wget")
        print("✓ wget detected - using resumable downloads")

    if not fast_downloaders:
        print("⚠ No fast downloaders found - using standard downloads")
        if sys.platform.startswith("linux"):
            print("  Install faster downloaders: sudo apt install aria2 axel wget")
        elif sys.platform.startswith("darwin"):  # macOS
            print("  Install faster downloaders via Homebrew:")
            print(
                "    brew install aria2      # Best option - multi-connection + resumable"
            )
            print("    brew install axel       # Good alternative - multi-connection")
            print("    brew install wget       # Basic resumable downloads")
    else:
        print(
            f"✓ Will try downloaders in order: {' -> '.join(fast_downloaders)} -> urllib"
        )

    # Check for RAR extractors availability
    rar_extractors = []

    if shutil.which("unrar"):
        rar_extractors.append("unrar")
        print("✓ unrar detected - can extract RAR files")

    if shutil.which("rar"):
        rar_extractors.append("rar")
        print("✓ rar detected - can extract RAR files")

    if shutil.which("unar"):
        rar_extractors.append("unar")
        print("✓ unar (The Unarchiver) detected - can extract RAR files")

    if shutil.which("dtrx"):
        rar_extractors.append("dtrx")
        print("✓ dtrx detected - universal archive extractor")

    # Check for Python rarfile
    try:
        import rarfile

        rar_extractors.append("rarfile")
        print("✓ Python rarfile detected - can extract RAR files")
    except ImportError:
        pass

    if not rar_extractors:
        print("⚠ No RAR extractors found")
        if sys.platform.startswith("darwin"):  # macOS
            print("  Install RAR extractor: brew install unar  # Recommended for Mac")
        elif sys.platform.startswith("linux"):
            print("  Install RAR extractor: sudo apt install unrar")
    else:
        print(f"✓ Will try RAR extractors: {' -> '.join(rar_extractors)}")

    print()

    # Clone and install SAM2
    sam2_success = clone_and_install_sam2()

    print()

    success_count = 0
    total_datasets = len(DATASETS)

    for filename, config in DATASETS.items():
        print(f"\n--- Processing {filename} ---")

        download_success = False
        extract_success = False

        # Handle download
        if Path(filename).exists():
            print(f"✓ {filename} already exists, skipping download")
            download_success = True
        else:
            print(f"Downloading {filename}...")
            if download_file(config["url"], filename):
                download_success = True
            else:
                print(f"✗ Failed to download {filename}")
                continue  # Skip extraction if download failed

        # Handle extraction (only if download was successful)
        if download_success:
            extract_dir = config["extract_dir"]
            if Path(extract_dir).exists() and not is_directory_empty(extract_dir):
                print(
                    f"✓ Extract directory {extract_dir}/ already exists and contains files, skipping extraction"
                )
                extract_success = True
            else:
                if Path(extract_dir).exists():
                    print(
                        f"⚠ Extract directory {extract_dir}/ exists but is empty, re-extracting..."
                    )
                else:
                    print(f"Extracting {filename} to {extract_dir}/...")

                if extract_file(filename, extract_dir):
                    extract_success = True
                else:
                    print(f"✗ Failed to extract {filename}")

        # Count as success only if both download and extraction succeeded
        if download_success and extract_success:
            success_count += 1
            print(f"✓ {filename} processed successfully")
        else:
            print(f"✗ {filename} processing incomplete")

    print(f"\n=== Setup Complete ===")
    print(f"SAM2: {'✓ Installed' if sam2_success else '✗ Failed'}")
    print(f"Datasets: Successfully processed {success_count}/{total_datasets}")

    if success_count == total_datasets and sam2_success:
        print("✓ All components ready!")
        return 0
    else:
        if not sam2_success:
            print("✗ SAM2 installation failed. Check the errors above.")
        if success_count < total_datasets:
            print("✗ Some datasets failed to process. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#==============================================================
#==============================================================
#==============================================================
# python script to install all conda pip from a saved txt file
#==============================================================

# run_installs.py ->>>>>>>>>>>>>>
# python run_installs.py requirements.txt
import subprocess
import sys
from pathlib import Path

def run(cmd: str):
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    steps_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("install_steps.txt")
    if not steps_file.exists():
        raise FileNotFoundError(f"Missing: {steps_file.resolve()}")

    for raw in steps_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Ensure pip uses the current interpreter (important in conda envs)
        if line.startswith("pip "):
            line = f"{sys.executable} -m {line}"

        run(line)

    print("\nAll install steps completed.")

if __name__ == "__main__":
    main()

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

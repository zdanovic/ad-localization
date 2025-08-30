import sys
from pathlib import Path

# Add src to sys.path for in-place execution without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env if present (optional)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from ai_loca.cli import main

if __name__ == "__main__":
    main()

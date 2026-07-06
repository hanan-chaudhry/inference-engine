import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent

tests = [
    root / "tests/unit/layernorm.py",
    root / "tests/unit/rmsnorm.py",
    root / "tests/unit/softmax.py",
    root / "tests/unit/attention.py",
    root / "tests/unit/multiAttention.py",
]

passed = 0
failed = 0

for test in tests:
    print(f"Running {test}...")
    result = subprocess.run([sys.executable, str(test)], cwd=root)
    if result.returncode == 0:
        passed += 1
    else:
        failed += 1
    print()

print("=" * 50)
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Total : {passed + failed}")
print("=" * 50)

sys.exit(0 if failed == 0 else 1)
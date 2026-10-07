import os
import sys
import tempfile
from pathlib import Path

# isolate every test run in its own data directory, before aerosculpt.config is imported
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="aerosculpt-test-")
os.environ["STRIPE_SECRET_KEY"] = ""
os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_test_secret"
os.environ["ALLOW_DEMO_BILLING"] = "1"
os.environ["OPEN_ACCESS"] = "0"   # the tests exercise real accounts
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

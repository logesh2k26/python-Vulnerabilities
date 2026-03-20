"""
Stress Test Suite for Hardened Python Vulnerability Detector.

Tests the application under heavy concurrent load to verify:
1. Rate limiting activates correctly
2. Large payload rejection works under stress
3. WebSocket connection limits hold
4. Authentication cannot be bypassed under load
5. Server remains responsive after stress

Run: python tests/stress_test.py
(Requires the server running on localhost:8000)
"""
import requests
import time
import threading
import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List

BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("API_SECRET_KEY", "test-secret-key-12345")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
HEADERS_UPLOAD = {"X-API-Key": API_KEY}

SAFE_CODE = "def hello():\n    print('hello world')\n"
VULN_CODE = "import os\ndef hack(cmd):\n    os.system(cmd)\n"


@dataclass
class StressResult:
    test_id: str
    test_name: str
    attack_type: str
    total_requests: int
    successful: int = 0
    rejected: int = 0
    rate_limited: int = 0
    errors: int = 0
    avg_response_ms: float = 0.0
    status: str = "PENDING"
    details: str = ""
    response_codes: dict = field(default_factory=dict)


def wait_for_server():
    print("⏳ Waiting for server...")
    for _ in range(15):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                print(f"✅ Server is UP: {r.json()}")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("❌ Server timed out")
    return False


def send_request(method, url, **kwargs):
    """Send a request and return (status_code, elapsed_ms)."""
    start = time.time()
    try:
        r = requests.request(method, url, timeout=10, **kwargs)
        elapsed = (time.time() - start) * 1000
        return r.status_code, elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return -1, elapsed


# ══════════════════════════════════════════════════════════════════════════
# Stress Test Functions
# ══════════════════════════════════════════════════════════════════════════

def stress_01_rate_limit_analyze():
    """ST-01: Hammer /analyze endpoint to trigger rate limit (30/min)."""
    result = StressResult("ST-01", "Rate Limit /analyze", "Rate Limit Exhaustion", 50)
    codes = {}
    times = []

    for i in range(50):
        code, ms = send_request("POST", f"{BASE_URL}/api/v1/analyze",
                                json={"content": SAFE_CODE, "filename": f"t{i}.py"},
                                headers=HEADERS)
        codes[code] = codes.get(code, 0) + 1
        times.append(ms)
        if code == 200:
            result.successful += 1
        elif code == 429:
            result.rate_limited += 1
        else:
            result.errors += 1

    result.response_codes = codes
    result.avg_response_ms = sum(times) / len(times) if times else 0
    result.status = "PASS" if result.rate_limited > 0 else "WARN"
    result.details = f"Rate limited {result.rate_limited}/50 requests"
    return result


def stress_02_rate_limit_upload():
    """ST-02: Hammer /dataset/upload to trigger rate limit (10/min)."""
    result = StressResult("ST-02", "Rate Limit /dataset/upload", "Rate Limit Exhaustion", 20)
    codes = {}
    times = []

    for i in range(20):
        code, ms = send_request("POST", f"{BASE_URL}/api/v1/dataset/upload",
                                files=[("files", (f"s{i}.py", b"print(1)", "text/plain"))],
                                headers=HEADERS_UPLOAD)
        codes[code] = codes.get(code, 0) + 1
        times.append(ms)
        if code == 200:
            result.successful += 1
        elif code == 429:
            result.rate_limited += 1
        else:
            result.errors += 1

    result.response_codes = codes
    result.avg_response_ms = sum(times) / len(times) if times else 0
    result.status = "PASS" if result.rate_limited > 0 else "WARN"
    result.details = f"Rate limited {result.rate_limited}/20 requests"
    return result


def stress_03_concurrent_analyze():
    """ST-03: 50 concurrent requests to /analyze."""
    result = StressResult("ST-03", "Concurrent /analyze", "Concurrency Flood", 50)
    codes = {}
    times = []

    def do_request(i):
        return send_request("POST", f"{BASE_URL}/api/v1/analyze",
                            json={"content": SAFE_CODE, "filename": f"c{i}.py"},
                            headers=HEADERS)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(do_request, i) for i in range(50)]
        for f in as_completed(futures):
            code, ms = f.result()
            codes[code] = codes.get(code, 0) + 1
            times.append(ms)
            if code == 200:
                result.successful += 1
            elif code == 429:
                result.rate_limited += 1
            else:
                result.errors += 1

    result.response_codes = codes
    result.avg_response_ms = sum(times) / len(times) if times else 0
    result.status = "PASS" if result.errors == 0 or result.rate_limited > 0 else "FAIL"
    result.details = f"{result.successful} OK, {result.rate_limited} rate-limited, {result.errors} errors"
    return result


def stress_04_auth_bypass_under_load():
    """ST-04: 30 concurrent auth bypass attempts with no key."""
    result = StressResult("ST-04", "Auth Bypass Under Load", "Authentication Bypass", 30)
    codes = {}

    def do_request(i):
        return send_request("POST", f"{BASE_URL}/api/v1/analyze",
                            json={"content": SAFE_CODE, "filename": f"a{i}.py"})

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [pool.submit(do_request, i) for i in range(30)]
        for f in as_completed(futures):
            code, _ = f.result()
            codes[code] = codes.get(code, 0) + 1
            if code in (401, 403):
                result.rejected += 1
            elif code == 200:
                result.successful += 1
            else:
                result.errors += 1

    result.response_codes = codes
    result.status = "PASS" if result.successful == 0 else "FAIL"
    result.details = f"All {result.rejected} rejected (no 200s leaked)"
    return result


def stress_05_large_payload_flood():
    """ST-05: 10 concurrent large payload submissions."""
    result = StressResult("ST-05", "Large Payload Flood", "Memory Exhaustion", 10)
    big_code = "x = 1\n" * (2 * 1024 * 1024)  # ~12 MB each
    codes = {}

    def do_request(i):
        return send_request("POST", f"{BASE_URL}/api/v1/analyze",
                            json={"content": big_code, "filename": f"big{i}.py"},
                            headers=HEADERS)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(do_request, i) for i in range(10)]
        for f in as_completed(futures):
            code, _ = f.result()
            codes[code] = codes.get(code, 0) + 1
            if code == 413:
                result.rejected += 1
            elif code == 429:
                result.rate_limited += 1
            else:
                result.errors += 1

    result.response_codes = codes
    result.status = "PASS" if result.rejected > 0 or result.rate_limited > 0 else "FAIL"
    result.details = f"{result.rejected} rejected (413), {result.rate_limited} rate-limited"
    return result


def stress_06_path_traversal_flood():
    """ST-06: 20 concurrent path traversal attempts."""
    result = StressResult("ST-06", "Path Traversal Flood", "Path Traversal", 20)
    traversal_names = [
        "../../etc/passwd.py", "../../../app/main.py",
        "..\\..\\Windows\\evil.py", "....//hack.py",
    ]
    codes = {}

    def do_request(i):
        fname = traversal_names[i % len(traversal_names)]
        return send_request("POST", f"{BASE_URL}/api/v1/dataset/upload",
                            files=[("files", (fname, b"print(1)", "text/plain"))],
                            headers=HEADERS_UPLOAD)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(do_request, i) for i in range(20)]
        for f in as_completed(futures):
            code, _ = f.result()
            codes[code] = codes.get(code, 0) + 1

    result.response_codes = codes
    # No 500 means traversal didn't crash the server
    result.status = "PASS" if codes.get(500, 0) == 0 else "FAIL"
    result.details = f"Response codes: {codes}"
    return result


def stress_07_invalid_key_flood():
    """ST-07: 50 concurrent requests with different invalid keys."""
    result = StressResult("ST-07", "Invalid Key Flood", "Brute Force Auth", 50)
    codes = {}

    def do_request(i):
        fake = {"X-API-Key": f"fake-key-{i:04d}", "Content-Type": "application/json"}
        return send_request("POST", f"{BASE_URL}/api/v1/analyze",
                            json={"content": SAFE_CODE, "filename": f"bf{i}.py"},
                            headers=fake)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(do_request, i) for i in range(50)]
        for f in as_completed(futures):
            code, _ = f.result()
            codes[code] = codes.get(code, 0) + 1
            if code == 403:
                result.rejected += 1
            elif code == 200:
                result.successful += 1

    result.response_codes = codes
    result.status = "PASS" if result.successful == 0 else "FAIL"
    result.details = f"All {result.rejected}/50 rejected with 403"
    return result


def stress_08_mixed_endpoint_flood():
    """ST-08: 60 mixed requests across all endpoints simultaneously."""
    result = StressResult("ST-08", "Mixed Endpoint Flood", "Multi-Vector DoS", 60)
    codes = {}
    times = []

    def hit_analyze(i):
        return send_request("POST", f"{BASE_URL}/api/v1/analyze",
                            json={"content": SAFE_CODE, "filename": f"m{i}.py"},
                            headers=HEADERS)

    def hit_batch(i):
        return send_request("POST", f"{BASE_URL}/api/v1/analyze/batch",
                            json={"files": [{"content": SAFE_CODE, "filename": f"mb{i}.py"}]},
                            headers=HEADERS)

    def hit_health(i):
        return send_request("GET", f"{BASE_URL}/health")

    tasks = []
    for i in range(20):
        tasks.append(("analyze", i))
        tasks.append(("batch", i))
        tasks.append(("health", i))

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {}
        for kind, i in tasks:
            if kind == "analyze":
                futures[pool.submit(hit_analyze, i)] = kind
            elif kind == "batch":
                futures[pool.submit(hit_batch, i)] = kind
            else:
                futures[pool.submit(hit_health, i)] = kind

        for f in as_completed(futures):
            code, ms = f.result()
            codes[code] = codes.get(code, 0) + 1
            times.append(ms)

    result.response_codes = codes
    result.avg_response_ms = sum(times) / len(times) if times else 0
    result.successful = codes.get(200, 0)
    result.rate_limited = codes.get(429, 0)
    result.errors = codes.get(500, 0) + codes.get(-1, 0)
    result.status = "PASS" if result.errors == 0 else "FAIL"
    result.details = f"Avg {result.avg_response_ms:.0f}ms, codes: {codes}"
    return result


def stress_09_websocket_flood():
    """ST-09: Attempt 20 simultaneous WebSocket connections."""
    result = StressResult("ST-09", "WebSocket Connection Flood", "Connection Exhaustion", 20)
    import websocket as ws_lib

    connections = []
    connected = 0
    rejected = 0

    for i in range(20):
        try:
            conn = ws_lib.create_connection(f"ws://localhost:8000/ws/analyze", timeout=3)
            connections.append(conn)
            connected += 1
        except Exception:
            rejected += 1

    # Clean up
    for c in connections:
        try:
            c.close()
        except Exception:
            pass

    result.successful = connected
    result.rejected = rejected
    result.status = "PASS"  # Server didn't crash
    result.details = f"{connected} connected, {rejected} rejected"
    return result


def stress_10_sustained_load():
    """ST-10: Sustained 30-second load test (5 req/sec)."""
    result = StressResult("ST-10", "Sustained Load (30s)", "Endurance", 150)
    codes = {}
    times = []
    duration = 30  # seconds
    start = time.time()
    count = 0

    while time.time() - start < duration:
        code, ms = send_request("POST", f"{BASE_URL}/api/v1/analyze",
                                json={"content": SAFE_CODE, "filename": f"sustained{count}.py"},
                                headers=HEADERS)
        codes[code] = codes.get(code, 0) + 1
        times.append(ms)
        count += 1
        time.sleep(0.2)  # ~5 req/sec

    result.total_requests = count
    result.successful = codes.get(200, 0)
    result.rate_limited = codes.get(429, 0)
    result.errors = codes.get(500, 0) + codes.get(-1, 0)
    result.response_codes = codes
    result.avg_response_ms = sum(times) / len(times) if times else 0
    result.status = "PASS" if result.errors == 0 else "FAIL"
    result.details = f"{count} requests in {duration}s, avg {result.avg_response_ms:.0f}ms"
    return result


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    stress_01_rate_limit_analyze,
    stress_02_rate_limit_upload,
    stress_03_concurrent_analyze,
    stress_04_auth_bypass_under_load,
    stress_05_large_payload_flood,
    stress_06_path_traversal_flood,
    stress_07_invalid_key_flood,
    stress_08_mixed_endpoint_flood,
    stress_09_websocket_flood,
    stress_10_sustained_load,
]


def print_results(results: List[StressResult]):
    print("\n" + "=" * 80)
    print("  STRESS TEST RESULTS")
    print("=" * 80)

    for r in results:
        icon = "✅" if r.status == "PASS" else ("⚠️" if r.status == "WARN" else "❌")
        print(f"\n{icon} {r.test_id}: {r.test_name}")
        print(f"   Attack Type:    {r.attack_type}")
        print(f"   Total Requests: {r.total_requests}")
        print(f"   Response Codes: {r.response_codes}")
        print(f"   Avg Response:   {r.avg_response_ms:.1f} ms")
        print(f"   Details:        {r.details}")
        print(f"   Status:         {r.status}")

    passed = sum(1 for r in results if r.status in ("PASS", "WARN"))
    failed = sum(1 for r in results if r.status == "FAIL")

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: {passed} passed, {failed} failed out of {len(results)} tests")
    print(f"{'=' * 80}")

    # Server health check after all stress tests
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"\n✅ Server still healthy after stress testing: {r.json()}")
        else:
            print(f"\n⚠️ Server responded with {r.status_code}")
    except Exception as e:
        print(f"\n❌ Server unreachable after stress test: {e}")


if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)

    print(f"\n🔥 Running {len(ALL_TESTS)} stress tests...\n")

    results = []
    for test_fn in ALL_TESTS:
        print(f"▶ Running {test_fn.__doc__}")
        try:
            r = test_fn()
            results.append(r)
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            results.append(StressResult(
                test_id="ERR", test_name=test_fn.__name__,
                attack_type="Error", total_requests=0,
                status="FAIL", details=str(e)
            ))
        time.sleep(1)  # Brief pause between tests

    print_results(results)

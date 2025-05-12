import requests
import time
import concurrent.futures
import argparse
from inference.inference_api import prepare_input  # Assuming this function prepares input data

def send_request(endpoint, input_data):
    try:
        response = requests.post(endpoint, json=input_data)
        return response.status_code, response.elapsed.total_seconds()
    except Exception as e:
        return None, str(e)

def run_load_test(endpoint, num_requests, concurrency, input_size):
    input_data = prepare_input(input_size)  # Prepare input data of the specified size
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, endpoint, input_data) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    success_count = sum(1 for status, _ in results if status == 200)
    failure_count = len(results) - success_count
    avg_response_time = sum(time for _, time in results if isinstance(time, float)) / len(results)

    print(f"Total Requests: {len(results)}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {failure_count}")
    print(f"Average Response Time: {avg_response_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a load test on the inference API.")
    parser.add_argument("--endpoint", required=True, help="Inference API endpoint URL")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--input-size", type=int, default=256, help="Size of input data")

    args = parser.parse_args()
    run_load_test(args.endpoint, args.num_requests, args.concurrency, args.input_size)

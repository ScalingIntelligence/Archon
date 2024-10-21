import time
import docker.models
import docker.models.containers
import docker.models.images
import requests
import docker
from pathlib import Path
from typing import List, Dict, Any
from docker.errors import DockerException, ImageNotFound

from .schema import ExecuteCodeResult

STARTUP_TIMEOUT_SECONDS = 20.0
PING_TIMEOUT_SECONDS = 1.0
WAIT_FOR_SERVER_BACKOFF_SECONDS = 1.0
IMAGE_NAME = "code-contests-python-execution-server"


class ExecutionError(Exception):
    """Custom exception for execution-related errors."""

    pass


class ExecutionServerClient:
    container: docker.models.containers.Container | None

    def __init__(self, port: int = 8005):
        """Initialize the ExecutionServerClient.

        Args:
            port (int): The port to run the execution server on.
        """
        self.port = port

        self.container = None

        self.base_url = f"http://localhost:{port}"
        self.docker_client = docker.from_env()
        self.dockerfile_path = Path(__file__).parent / "execution_server.Dockerfile"

    def __enter__(self):
        """Start the Docker container and wait for the server to be ready."""
        try:
            print("Starting docker")
            image = self._get_image()
            self.container = self.docker_client.containers.run(
                image=image,
                detach=True,
                ports={f"{self.port}/tcp": self.port},
                auto_remove=True,
            )
            print("waiting for server")
            self._wait_for_server(STARTUP_TIMEOUT_SECONDS)

            return self
        except:
            self.stop_container()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the Docker container."""
        self.stop_container()

    def stop_container(self):
        if self.container is not None:
            self.container.stop()

            self.container = None

    def _get_image(self) -> docker.models.images.Image:
        """Check if the Docker image exists, and build it if it doesn't."""
        try:
            image = self.docker_client.images.get(IMAGE_NAME)
        except ImageNotFound:
            print(f"Image '{IMAGE_NAME}' not found. Building...")
            image = self._build_new_image()

        return image

    def _build_new_image(self) -> docker.models.images.Image:
        """Build the Docker image from the Dockerfile."""
        if not self.dockerfile_path.exists():
            raise ExecutionError(f"Dockerfile not found at {self.dockerfile_path}")

        try:
            image, _ = self.docker_client.images.build(
                dockerfile=self.dockerfile_path, path=".", tag=IMAGE_NAME
            )
            print(f"Image '{IMAGE_NAME}' built successfully.")
        except DockerException as e:
            raise ExecutionError(f"Failed to build Docker image: {str(e)}")

        return image

    def execute_code(
        self,
        code: str,
        input_expected_output_pairs: List[str],
        timeout: float,
        memory_limit_bytes: int,
    ) -> bool:
        """
        Execute the given code with the provided inputs.

        Args:
            code (str): The Python code to execute.
            input_expected_output_pairs (List[Tuple[str, str]]): List of input/expected output strings for the code.
            timeout (float): Maximum execution time for each input.
            memory_limit_bytes (int): memory limit of the program.

        Returns:
            bool: indicates if the code passed the tests.

        Raises:
            ExecutionError: If there's an error during execution or communication with the server.
        """
        print("executing code")
        try:
            response = requests.post(
                f"{self.base_url}/execute",
                json={
                    "code": code,
                    "input_expected_output_pairs": input_expected_output_pairs,
                    "timeout": timeout,
                    "memory_limit_bytes": memory_limit_bytes,
                },
            )
        except requests.RequestException as e:
            raise ExecutionError(
                f"Failed to communicate with execution server: {str(e)}"
            )

        if response.status_code != 200:
            raise ExecutionError(f"Execution failed with status {response.status_code}")

        return ExecuteCodeResult(**response.json()).correct

    def ping(self) -> bool:
        """Check if the server is responsive.

        Returns:
            bool: True if the server responds with "pong", False otherwise.
        """
        try:
            response = requests.get(
                f"{self.base_url}/ping", timeout=PING_TIMEOUT_SECONDS
            )
            return response.text == '"pong"'
        except requests.RequestException:
            return False

    def _wait_for_server(self, timeout: float) -> None:
        """Internal method to wait for the server to be ready.

        Args:
            timeout (float): Maximum time to wait for the server to be ready.

        Raises:
            ExecutionError: If the server doesn't respond within the timeout period.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ping():
                return
            time.sleep(WAIT_FOR_SERVER_BACKOFF_SECONDS)
        raise ExecutionError(
            "Execution server failed to start within the specified timeout"
        )

import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from divi.interfaces import CircuitRunner


class SlurmService(CircuitRunner):
    def __init__(
        self,
        shots: int = 1024,
        base_dir: str = "/tmp/divi_jobs",
        simulator_exec: str = "python slurm_simulate.py",
    ):
        super().__init__(shots)
        self.base_dir = Path(base_dir)
        self.simulator_exec = simulator_exec
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.n_processes = 2

    def _write_batch_job(self, job_id: str, circuits: list[tuple[str, str]]) -> Path:
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        circuit_map = {}
        for i, circuit in enumerate(circuits):
            circuit_label, circuit_qasm = circuit
            circuit_file = job_dir / f"circuit_{circuit_label}.qasm"
            circuit_file.write_text(circuit_qasm)
            circuit_map[i] = circuit_label

        # Save the mapping so results can be loaded later
        with open(job_dir / "circuit_map.json", "w") as f:
            json.dump(circuit_map, f)

            # Write SLURM script
        run_sh = job_dir / "run.sh"

        script = f"""
        #!/bin/bash
        #SBATCH --job-name=divi_batch
        #SBATCH --output={job_dir}/slurm_output.log
        #SBATCH --ntasks=1
        #SBATCH --time=00:10:00
        #SBATCH --mem=16G
        #
        {self.simulator_exec} --input-dir {job_dir} --shots {self.shots} --n-processes {self.n_processes}
        """
        # """

        run_sh.write_text(script.strip())
        run_sh.chmod(0o755)

        return run_sh

    def submit_circuits(self, circuits: dict[str, str], tag: str = "default", **kwargs):
        job_id = str(uuid.uuid4())
        run_sh = self._write_batch_job(job_id, circuits)

        result = subprocess.run(["sbatch", str(run_sh)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")
        return job_id

    def wait_for_completion(
        self,
        job_id: str,
        labels: list[str],
        timeout: int = 3600,
        poll_interval: int = 10,
    ):
        """
        Wait for SLURM job completion signaled by done.flag and result files.
        """
        start_time = time.time()
        done_flag = self.base_dir / job_id / "done.flag"
        job_dir = self.base_dir / job_id
        while time.time() - start_time < timeout:
            if done_flag.exists():
                # Confirm all expected result files are present and readable
                missing = [
                    i
                    for i in range(len(labels))
                    if not (job_dir / f"result_{labels[i]}.json").exists()
                ]
                if not missing:
                    return True
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds.")

    def get_job_results(self, job_id: str, on_complete=None):
        job_dir = self.base_dir / job_id
        if not job_dir.exists():
            raise ValueError(f"Job {job_id} does not exist.")

        results = []
        for result_file in job_dir.glob("result_*.json"):
            label = result_file.stem.split("_", 1)[1]
            result = {"label": label}
            with open(result_file) as f:
                data = json.load(f)
            result["results"] = data.get("counts", data)
            results.append(result)

        if on_complete:
            on_complete(results)

        return results

    def cleanup_job(self, job_id: str):
        """
        Delete the job directory and all its contents for the given job_id.
        """
        job_dir = self.base_dir / job_id
        if job_dir.exists() and job_dir.is_dir():
            for item in job_dir.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    # Recursively delete subdirectories if any
                    shutil.rmtree(item)
            job_dir.rmdir()

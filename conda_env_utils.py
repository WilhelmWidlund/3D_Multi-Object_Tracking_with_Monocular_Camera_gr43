import subprocess


def handle_conda_environment(env_name, requirements_file, python_version):
    def create_conda_environment():
        subprocess.run(f"conda create --name {env_name} python={python_version} --file {requirements_file}", shell=True)
        print(f"Conda environment {env_name} created.")

    def switch_conda_environment():
        subprocess.run(f"conda activate {env_name}", shell=True)
        print(f"Switched to conda environment: {env_name}")

    env_exists = False
    try:
        switch_conda_environment()
        env_exists = True
    except subprocess.CalledProcessError as e:
        pass

    if not env_exists:
        create_conda_environment()
    else:
        print(f"Conda environment {env_name} already exists.")


# Example usage:
handle_conda_environment("env1", "requirements.txt", "3.9")
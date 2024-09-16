import subprocess

subprocess.run(["git", "submodule", "init"])
subprocess.run(["git", "submodule", "update", '--recursive'])
subprocess.run(['pip', 'install', '-e', '.'], cwd='./src/real_esrgan_generator/extern/Real_ESRGAN/')
with open('src/real_esrgan_generator/extern/Real_ESRGAN/__init__.py', 'w') as export:
    export.write("")

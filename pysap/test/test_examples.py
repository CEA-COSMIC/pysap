import os
import subprocess

currentdir = os.path.dirname(__file__)
examplesdir = os.path.join(
    currentdir, os.path.join(os.pardir, os.pardir), 'examples'
)

example_files = []
for root, dirs, files in os.walk(examplesdir):
    for basneame in files:
        if basneame.endswith('.py'):
            example_files.append(os.path.abspath(
                os.path.join(root, basneame)
            ))
print('"{0}" examples found!'.format(len(example_files)))

for path in example_files:
    print('-- ', path)
    cmd = ['python3', path]
    subprocess.check_call(cmd, env=os.environ)

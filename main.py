import subprocess
from collections import OrderedDict


if __name__ == '__main__':
    # creating the args
    args = OrderedDict(dset='USPS',
                       limit=250)

    # creating the command
    command = 'python train.py'
    for arg in args:
        command += ' --{} {}'.format(arg, args[arg])

    # running the command
    print(command)
    subprocess.call(command.split(' '), shell=True)
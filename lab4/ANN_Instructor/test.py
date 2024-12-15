import sys

try:
    arg1 = sys.argv[1]  # First argument
    arg2 = sys.argv[2]  # Second argument

    print(arg1+arg2)

except:
    print('run as `python3 test.py arg1 arg2`')

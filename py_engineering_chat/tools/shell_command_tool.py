# Print stdout and stderr
print("\033[94mCommand output:\033[0m")
print(result.stdout)
if result.stderr.strip():
    print("\033[91mCommand errors:\033[0m")
    print(result.stderr)
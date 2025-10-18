import re, math, pathlib

file = pathlib.Path("stats_summary.txt")          # adjust path if needed
txt  = file.read_text()

mean_diff = float(re.search(r"mean diff\s+([-0-9.]+)", txt).group(1))
t_val     = float(re.search(r"t\s*=\s*([-0-9.]+)",     txt).group(1))
n         = int(  re.search(r"\(n=(\d+)\)",            txt).group(1))

sem = abs(mean_diff / t_val)
sd  = sem * math.sqrt(n)

print(f"n   = {n}")
print(f"SD  = {sd:.4f}")
print(f"SEM = {sem:.4f}")
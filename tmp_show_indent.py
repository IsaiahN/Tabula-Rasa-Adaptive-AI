import io
p='src/arc_integration/continuous_learning_loop.py'
with open(p, 'r', encoding='utf-8') as f:
    lines=f.readlines()
for i in range(6355, 6410):
    print(i+1, repr(lines[i]))

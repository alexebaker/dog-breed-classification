ids = []

with open('data/classification.csv', 'r') as f:
    for line in f.readlines()[1:]:
        ids.append(line.strip().split(',')[0])

dups = []
for idx in ids:
    if ids.count(idx) > 1:
        dups.append(idx)


print(dups)
print(len(dups))

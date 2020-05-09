
# More readable, inefficient, old impl
idxs = []
assert last_idx + 1 > batch_size, "Not enough transitions"
while len(idxs) < batch_size:
    i = np.random.randint(0, last_idx)
    if i in idxs:
        continue
    if np.all(self.not_dones[i:i + T] == 1.):
        idxs.append(i)
idxs = np.array(idxs)

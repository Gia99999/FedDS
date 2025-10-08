def add_noise(y, noise_ratio):
    y = y.clone()
    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)
    noise_indices = np.random.choice(n_samples, n_noisy, replace=False)
    y_noisy = y.clone()
    unique_labels = y.unique().numpy()
    for i in noise_indices:
        y_noisy[i] = np.random.choice(np.setdiff1d(unique_labels, y_noisy[i].item()))
    return y_noisy
def add_asymmetric_noise(y, noise_ratio):
    y_noisy = y.clone()
    # CIFAR-10 classes: 0:airplane, 1:automobile, 2:bird, 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck
    noise_map = {
        9: 1,  # Truck -> Automobile
        2: 0,  # Bird -> Airplane
        4: 7,  # Deer -> Horse
        3: 5   # Cat -> Dog
    }
    
    source_classes = list(noise_map.keys())
    
    for src_cls, tgt_cls in noise_map.items():
        src_indices = (y == src_cls).nonzero(as_tuple=True)[0]
        n_samples_to_flip = int(len(src_indices) * noise_ratio)
        
        if n_samples_to_flip > 0:
            flip_indices = np.random.choice(src_indices.cpu().numpy(), n_samples_to_flip, replace=False)
            y_noisy[flip_indices] = tgt_cls
            
    return y_noisy

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset_full = datasets.MNIST('../data/mnist', train=True, download=True, transform=transform_train)
test_dataset_full = datasets.MNIST('../data/mnist', train=False, download=True, transform=transform_test)

X_train_full = train_dataset_full.data.unsqueeze(1).float() / 255.0
y_train_full = train_dataset_full.targets

X_test_full = test_dataset_full.data.unsqueeze(1).float() / 255.0
y_test_full = test_dataset_full.targets


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_dataset_full = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=transform_train)
test_dataset_full = datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=transform_test)

X_train_full = torch.tensor(train_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_train_full = torch.tensor(train_dataset_full.targets)
X_test_full = torch.tensor(test_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_test_full = torch.tensor(test_dataset_full.targets)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


train_dataset_full = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=transform_train)
test_dataset_full = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=transform_test)

X_train_full = torch.tensor(train_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_train_full = torch.tensor(train_dataset_full.targets)
X_test_full = torch.tensor(test_dataset_full.data).permute(0, 3, 1, 2).float() / 255.0
y_test_full = torch.tensor(test_dataset_full.targets)



def prepare_datasets(train_data, train_targets, test_data, test_targets, num_users, public_ratio=0.1):
    X_train_client, X_public, y_train_client, y_public = train_test_split(
        train_data, train_targets, test_size=public_ratio, stratify=train_targets, random_state=42
    )
    num_items = int(len(X_train_client) / num_users)
    dict_users = {}
    all_idxs = list(range(len(X_train_client)))
    for i in range(num_users):
        chosen = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = set(chosen)
        all_idxs = list(set(all_idxs) - dict_users[i])
    if len(all_idxs) > 0:
        dict_users[num_users-1] = dict_users[num_users-1].union(all_idxs)
    dataset_clients = (X_train_client, y_train_client, dict_users)
    dataset_public = torch.utils.data.TensorDataset(X_public, y_public)
    dataset_test = torch.utils.data.TensorDataset(test_data, test_targets)
    return dataset_clients, dataset_public, dataset_test

dataset_clients, dataset_public, dataset_test = prepare_datasets(
    X_train_full, y_train_full, X_test_full, y_test_full, args.num_users, public_ratio=0.1
)

X_train_client, y_train_client, dict_users_train = dataset_clients
public_data_loader = DataLoader(dataset_public, batch_size=args.bs, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False)


def prepare_datasets_dirichlet(train_data, train_targets, test_data, test_targets, num_users, alpha=10, public_ratio=0.1):
    X_train_client, X_public, y_train_client, y_public = train_test_split(
        train_data, train_targets, test_size=public_ratio, stratify=train_targets, random_state=42
    )

    y_train_np = y_train_client.numpy()
    idxs = np.arange(len(y_train_np))
    num_classes = len(np.unique(y_train_np))

    min_size = 0
    while min_size < 10:  
        idx_batch = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = idxs[y_train_np == k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(alpha * np.ones(num_users))
            proportions = np.array([p * (len(idx_j) < len(y_train_np) / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split_idx = np.split(idx_k, proportions)
            for i, idx in enumerate(split_idx):
                idx_batch[i].extend(idx)
        min_size = min(len(idx_j) for idx_j in idx_batch)

    dict_users = {i: set(idx_batch[i]) for i in range(num_users)}

    dataset_clients = (X_train_client, y_train_client, dict_users)
    dataset_public = torch.utils.data.TensorDataset(X_public, y_public)
    dataset_test = torch.utils.data.TensorDataset(test_data, test_targets)
    return dataset_clients, dataset_public, dataset_test


dataset_clients, dataset_public, dataset_test = prepare_datasets_dirichlet(
    X_train_full, y_train_full, X_test_full, y_test_full, args.num_users, alpha=10, public_ratio=0.1
)

X_train_client, y_train_client, dict_users_train = dataset_clients
public_data_loader = DataLoader(dataset_public, batch_size=args.bs, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False)

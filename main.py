num_runs = 3
all_runs_results = {name: [] for name in ds_variants.keys()}

class LocalUpdate(object):
    def __init__(self, args, noisy_data):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.noisy_data = noisy_data
        self.ldr_train = self.get_noisy_loader()

    def get_noisy_loader(self):
        images, labels = self.noisy_data
        dataset_noisy = torch.utils.data.TensorDataset(images, labels)
        return DataLoader(dataset_noisy, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, w_glob):
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        global_params = copy.deepcopy(w_glob)

        for _ in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # FedProx proximal term
                prox_reg = 0.0
                mu = 0.05  # FedProx 正则化系数，可调
                for name, param in net.state_dict().items(): 
                    g_param = global_params[name].detach().to(self.args.device)
                    prox_reg += ((param - g_param) ** 2).sum()
                loss += (mu / 2) * prox_reg

                loss.backward()
                optimizer.step()
        return net.state_dict(), loss.item()

def test_img(net_g, data_loader, args):
    net_g.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            pred = log_probs.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    accuracy = 100.0 * correct / len(data_loader.dataset)
    return accuracy

for run in range(num_runs):
    seed = run
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n------------- Starting Run {run+1}/{num_runs} (Training Seed: {seed}) -------------")

    noisy_train_data = {}
    for idx in range(args.num_users):
        train_idxs = list(dict_users_train[idx])
        X_train_client_part = X_train_client[train_idxs]
        y_train_client_part = y_train_client[train_idxs]
        
        noise_ratio = client_noises[idx] 
        
        y_train_noisy = add_noise(y_train_client_part, noise_ratio)
        noisy_train_data[idx] = (X_train_client_part, y_train_noisy)
        
    for variant_name, ds_func in ds_variants.items():
        print(f"\nRunning variant: {variant_name}")
        net_glob_variant = CNNCifar(args=args).to(args.device)
        w_locals_variant = [net_glob_variant.state_dict() for _ in range(args.num_users)]
        acc_list = []
        for round_num in range(args.epochs):
            print(f"  Round {round_num+1}/{args.epochs}")
            client_predictions = []
            
            w_locals_round = []

            m = max(int(args.frac * args.num_users), 1)
            selected_clients = np.random.choice(range(args.num_users), m, replace=False)
            
            for idx in selected_clients:
                local = LocalUpdate(args=args, noisy_data=noisy_train_data[idx])
                w, _ = local.train(net=copy.deepcopy(net_glob_variant).to(args.device))
                w_locals_round.append(w)
                
                net_client = CNNCifar(args=args).to(args.device)
                net_client.load_state_dict(w)
                net_client.eval()
                preds = []
                with torch.no_grad():
                    for data, _ in public_data_loader:
                        data = data.to(args.device)
                        outputs = net_client(data)
                        _, predicted = torch.max(outputs, 1)
                        preds.extend(predicted.cpu().numpy())
                client_predictions.append(preds)
            
            client_predictions_transposed = list(zip(*client_predictions))

            if ds_func is None:
                w_glob_new, _ = FedAvg(w_locals_round, use_equal_weights=True)
            else:
                if not client_predictions_transposed:
                    w_glob_new = net_glob_variant.state_dict()
                else:
                    _, confusion_matrices = ds_func(client_predictions_transposed, num_classes=args.num_classes)
                    
                    client_weights_adjusted = None
                    if confusion_matrices is not None:
                        client_accuracies = [np.mean(np.diag(cm)) for cm in confusion_matrices]
                        client_weights_adjusted = np.array(client_accuracies)

                    w_glob_new, _ = FedAvg(w_locals_round,
                                          weights=client_weights_adjusted,
                                          use_equal_weights=False)

            net_glob_variant.load_state_dict(w_glob_new)
            acc = test_img(net_glob_variant, test_loader, args)
            print(f"    Global Test Accuracy: {acc:.2f}%")
            acc_list.append(acc.item())
        
        all_runs_results[variant_name].append(acc_list)

final_reported_accuracy = {}

print("\n\n" + "="*60)
print(" " * 20 + "FINAL RESULTS SUMMARY")
print("="*60)

for variant_name in ds_variants.keys():

    last_10_rounds_accs = [run_results[-10:] for run_results in all_runs_results[variant_name]]
    avg_acc_per_run = np.mean(last_10_rounds_accs, axis=1)
    final_avg = np.mean(avg_acc_per_run)
    final_std = np.std(avg_acc_per_run)    
    final_reported_accuracy[variant_name] = (final_avg, final_std)
    
    print(f"Variant: {variant_name}")
    print(f"  - Last 10 rounds average accuracy for each run: {[f'{acc:.2f}%' for acc in avg_acc_per_run]}")
    print(f"  - Reported Accuracy (Mean ± Std Dev over {num_runs} runs): {final_avg:.2f}% ± {final_std:.2f}%")
    print("-" * 60)


plt.figure(figsize=(14, 8))
for variant_name in ds_variants.keys():
    avg_accs_across_runs = np.mean(all_runs_results[variant_name], axis=0)
    final_avg, final_std = final_reported_accuracy[variant_name]
    label_text = f"{variant_name} (Final Acc: {final_avg:.2f}%)"
    plt.plot(range(1, args.epochs + 1), avg_accs_across_runs, label=label_text, marker='o', linestyle='--', markersize=4)

plt.xlabel("Communication Rounds")
plt.ylabel("Global Test Accuracy (%)")
plt.legend()
plt.title(f"Comparison of Methods (CIFAR-10, Avg over {num_runs} runs, Fixed Random Noise)")
plt.grid(True)
plt.show()

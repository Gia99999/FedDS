def FedAvg(w, weights=None, use_equal_weights=False):
    w_avg = copy.deepcopy(w[0])
    if use_equal_weights or weights is None or len(w) != len(weights):
        total_weight = len(w)
        normalized_weights = [1.0 / total_weight] * len(w)
    else:
        total_weight = sum(weights)
        if total_weight == 0: # Fallback to equal weights if all weights are zero
            total_weight = len(w)
            normalized_weights = [1.0 / total_weight] * len(w)
        else:
            normalized_weights = [weight / total_weight for weight in weights]

    for k in w_avg.keys():
        w_avg[k] = normalized_weights[0] * w[0][k]
        for i in range(1, len(w)):
            w_avg[k] += normalized_weights[i] * w[i][k]
    return w_avg, normalized_weights
  
  def dawid_skene_standard(predictions, num_classes, max_iter=100, tol=1e-5):
    num_items = len(predictions)
    num_annotators = len(predictions[0])
    true_label_probs = np.zeros((num_items, num_classes))
    for i in range(num_items):
        for c in range(num_classes):
            true_label_probs[i, c] = np.mean([1 if p == c else 0 for p in predictions[i]])
    true_label_probs += 1e-10
    true_label_probs /= true_label_probs.sum(axis=1, keepdims=True)

    confusion_matrices = np.zeros((num_annotators, num_classes, num_classes))
    for a in range(num_annotators):
        for i in range(num_items):
            pred = predictions[i][a]
            for c in range(num_classes):
                confusion_matrices[a][c][pred] += true_label_probs[i, c]
        row_sums = confusion_matrices[a].sum(axis=1, keepdims=True)
        row_sums[row_sums==0] = 1
        confusion_matrices[a] /= row_sums

    for it in range(max_iter):
        prev_true_label_probs = true_label_probs.copy()
        for i in range(num_items):
            for c in range(num_classes):
                prod = 1.0
                for a in range(num_annotators):
                    pred = predictions[i][a]
                    prod *= confusion_matrices[a][c][pred]
                true_label_probs[i, c] = prod
            total = true_label_probs[i].sum()
            if total > 0:
                true_label_probs[i] /= total
            else:
                true_label_probs[i] = np.ones(num_classes) / num_classes
        
        new_confusion_matrices = np.zeros_like(confusion_matrices)
        for a in range(num_annotators):
            for i in range(num_items):
                pred = predictions[i][a]
                for c in range(num_classes):
                    new_confusion_matrices[a][c][pred] += true_label_probs[i, c]
            row_sums = new_confusion_matrices[a].sum(axis=1, keepdims=True)
            row_sums[row_sums==0] = 1
            new_confusion_matrices[a] /= row_sums
        confusion_matrices = new_confusion_matrices
        if np.max(np.abs(true_label_probs - prev_true_label_probs)) < tol:
            break
    estimated_labels = np.argmax(true_label_probs, axis=1)
    return estimated_labels, confusion_matrices

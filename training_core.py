def federated_training(model_type="plain"):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    import csv
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from shared import update_progress, append_metric, get_progress

    # Config
    NUM_CLIENTS = 5
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    def partition_dataset(dataset, num_clients):
        data_per_client = len(dataset) // num_clients
        return [list(range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]

    client_indices = partition_dataset(trainset, NUM_CLIENTS)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*8*8, 128), nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.conv(x)
            return self.fc(x)

    class SimpleCNNWithDropout(SimpleCNN):
        def __init__(self):
            super().__init__()
            self.conv.insert(3, nn.Dropout(0.25))
            self.conv.append(nn.Dropout(0.25))
            self.fc.insert(2, nn.Dropout(0.5))

    def train_local(model, dataloader, epochs, use_dp=False):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        if use_dp:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=0.8,
                max_grad_norm=1.0
            )

        model.train()
        for _ in range(epochs):
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        return model._module.state_dict() if use_dp else model.state_dict()

    def average_weights(weights):
        avg = copy.deepcopy(weights[0])
        for k in avg:
            for i in range(1, len(weights)):
                avg[k] += weights[i][k]
            avg[k] /= len(weights)
        return avg

    def evaluate_model(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        return 100 * correct / total

    def collect_mia_data(model, loader, label, max_samples=300):
        model.eval()
        feats, labels = [], []
        with torch.no_grad():
            count = 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                prob = torch.softmax(out, dim=1)
                loss = nn.CrossEntropyLoss(reduction='none')(out, y)
                for p, l in zip(prob, loss):
                    if count >= max_samples: break
                    feats.append(torch.cat([p, l.unsqueeze(0)]).cpu().numpy())
                    labels.append(label)
                    count += 1
        return feats, labels

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def train_mia_classifier(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        return clf

    update_progress("total_rounds", NUM_ROUNDS)
    update_progress("status", "training")
    model_class = SimpleCNN if model_type == "plain" else SimpleCNNWithDropout if model_type == "dropout" else SimpleCNN
    use_dp = model_type == "dp"

    global_model = model_class().to(DEVICE)
    client0_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, client_indices[0]), batch_size=32, shuffle=True)
    mia_test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    for rnd in range(NUM_ROUNDS):
        update_progress("current_round", rnd + 1)
        weights = []
        for c in range(NUM_CLIENTS):
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, client_indices[c]), batch_size=BATCH_SIZE, shuffle=True)
            local_model = copy.deepcopy(global_model).to(DEVICE)
            w = train_local(local_model, loader, LOCAL_EPOCHS, use_dp)
            weights.append(w)

        global_model.load_state_dict(average_weights(weights))

        acc = evaluate_model(global_model, test_loader)
        append_metric("accuracy", acc)

        fin, lin = collect_mia_data(global_model, client0_loader, 1)
        fout, lout = collect_mia_data(global_model, mia_test_loader, 0)
        X, y = np.array(fin + fout), np.array(lin + lout)
        clf = train_mia_classifier(X, y)
        auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        append_metric("mia_auc", auc * 100)

    update_progress("status", "done")
        # Sauvegarde des courbes après le dernier round

    rounds = list(range(1, NUM_ROUNDS + 1))

    # Courbe Accuracy et MIA AUC
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, get_progress()["accuracy"], marker='o', label="Test Accuracy (%)")
    plt.plot(rounds, get_progress()["mia_auc"], marker='x', label="MIA AUC (%)")
    plt.xlabel("Federated Round")
    plt.ylabel("Metric (%)")
    plt.title("Test Accuracy & MIA AUC per Round")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/miacurve.png")  # <- Sauvegarde ici
    plt.close()

    # Courbe ROC (MIA finale)
    y_scores = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_scores)
    auc = roc_auc_score(y, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final MIA ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/miaroc.png")  # <- Sauvegarde ici
    plt.close()

    update_progress("message", f"Training completed using {model_type} model.")
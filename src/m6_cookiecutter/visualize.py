import io
import matplotlib.pyplot as plt
import torch
import typer
from m6_cookiecutter.model import MyAwesomeModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    model: torch.nn.Module = MyAwesomeModel().to(DEVICE)
    
    # Load model checkpoint with pre-loaded buffer
    with open(model_checkpoint, "rb") as f:
        buffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(buffer, map_location=DEVICE))
    model.eval()
    model.fc = torch.nn.Identity()

    # Load test data with pre-loaded buffers
    with open("data/processed/test_images.pt", "rb") as f:
        test_images = torch.load(io.BytesIO(f.read()), map_location=DEVICE)
    with open("data/processed/test_target.pt", "rb") as f:
        test_target = torch.load(io.BytesIO(f.read()), map_location=DEVICE)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)
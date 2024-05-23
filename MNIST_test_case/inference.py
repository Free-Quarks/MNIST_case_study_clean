import torch
import torchvision.datasets as datasets  # Standard datasets
from torchvision.utils import save_image
from torchvision import transforms

# performing inference to generate new images
def simple_inference(model_load, dataset, digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    # collects 10 examples of the given digit
    images = []
    idx = 0
    for x, y in dataset:
        if y == digit:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    # generates mu and logvar for each image
    encodings_digit = []
    for d in range(len(images)):
        with torch.no_grad():
            mu, logvar = model_load.encode(images[d].view(1, 784))
        encodings_digit.append((mu, logvar))

    # samples one from the list to generate similar ones
    mu, logvar = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(mu)
        z = mu + torch.exp(0.5*logvar) * epsilon
        out = model_load.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"./MNIST_test_case/generated_images/simple_generated_{digit}_ex{example}.png")

if __name__ == "__main__":
    # load model
    model_load = torch.load('./MNIST_test_case/saved_models/vae_model.pt')
    model_load.eval() 
    # load dataset
    dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)

    for idx in range(10):
        simple_inference(model_load, dataset, idx, num_examples=2)